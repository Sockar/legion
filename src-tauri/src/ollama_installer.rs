use std::{
    env,
    path::{Path, PathBuf},
    time::Duration,
};

use futures_util::StreamExt;
use reqwest::{header::ACCEPT, Client, Url};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tauri::AppHandle;
#[cfg(any(target_os = "macos", target_os = "linux"))]
use tauri::Manager;
use tokio::{
    fs::{self, File},
    io::AsyncWriteExt,
    process::Command,
};

const RELEASE_API: &str = "https://api.github.com/repos/ollama/ollama/releases/latest";
const MAX_DOWNLOAD_BYTES: u64 = 3 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InstallerTarget {
    WindowsX64,
    Mac,
    LinuxX64,
    LinuxArm64,
}

impl InstallerTarget {
    fn asset_name(self) -> &'static str {
        match self {
            Self::WindowsX64 => "OllamaSetup.exe",
            Self::Mac => "Ollama-darwin.zip",
            Self::LinuxX64 => "ollama-linux-amd64.tar.zst",
            Self::LinuxArm64 => "ollama-linux-arm64.tar.zst",
        }
    }

    fn expected_content_type(self, content_type: &str) -> bool {
        let content_type = content_type
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        match self {
            Self::WindowsX64 => content_type == "application/octet-stream",
            Self::Mac => {
                content_type == "application/zip" || content_type == "application/octet-stream"
            }
            Self::LinuxX64 | Self::LinuxArm64 => {
                content_type == "application/zstd" || content_type == "application/octet-stream"
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<ReleaseAsset>,
}

#[derive(Debug, Deserialize)]
struct ReleaseAsset {
    name: String,
    size: u64,
    digest: Option<String>,
    browser_download_url: String,
}

fn installer_target(os: &str, arch: &str) -> Result<InstallerTarget, String> {
    match (os, arch) {
        ("windows", "x86_64") => Ok(InstallerTarget::WindowsX64),
        ("macos", "x86_64" | "aarch64") => Ok(InstallerTarget::Mac),
        ("linux", "x86_64") => Ok(InstallerTarget::LinuxX64),
        ("linux", "aarch64") => Ok(InstallerTarget::LinuxArm64),
        ("windows", _) => Err(format!(
            "Automatic Ollama installation is not supported for Windows architecture '{arch}'."
        )),
        ("macos", _) => Err(format!(
            "Automatic Ollama installation is not supported for macOS architecture '{arch}'."
        )),
        ("linux", _) => Err(format!(
            "Automatic Ollama installation is not supported for Linux architecture '{arch}'."
        )),
        _ => Err(format!(
            "Automatic Ollama installation is not supported on '{os}'."
        )),
    }
}

fn validate_release_asset(target: InstallerTarget, asset: &ReleaseAsset) -> Result<Url, String> {
    if asset.name != target.asset_name() {
        return Err(format!(
            "Ollama release did not contain the expected {} asset.",
            target.asset_name()
        ));
    }
    if asset.size < 100 * 1024 * 1024 || asset.size > MAX_DOWNLOAD_BYTES {
        return Err(format!(
            "The published Ollama installer size ({} bytes) is outside the allowed range.",
            asset.size
        ));
    }
    let url = Url::parse(&asset.browser_download_url)
        .map_err(|error| format!("Invalid Ollama release URL: {error}"))?;
    if url.scheme() != "https"
        || url.host_str() != Some("github.com")
        || !url.path().starts_with("/ollama/ollama/releases/download/")
    {
        return Err(
            "The Ollama release asset did not use the expected HTTPS GitHub URL.".to_owned(),
        );
    }
    if let Some(digest) = &asset.digest {
        let Some(hash) = digest.strip_prefix("sha256:") else {
            return Err("The Ollama release published an unsupported checksum format.".to_owned());
        };
        if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("The Ollama release published an invalid SHA-256 checksum.".to_owned());
        }
    }
    Ok(url)
}

async fn download_release_asset(
    client: &Client,
    target: InstallerTarget,
    asset: &ReleaseAsset,
    destination: &Path,
) -> Result<bool, String> {
    let url = validate_release_asset(target, asset)?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|error| format!("Could not download Ollama: {error}"))?
        .error_for_status()
        .map_err(|error| format!("Could not download Ollama: {error}"))?;
    let final_url = response.url();
    if final_url.scheme() != "https"
        || !matches!(
            final_url.host_str(),
            Some(
                "github.com"
                    | "release-assets.githubusercontent.com"
                    | "objects.githubusercontent.com"
            )
        )
    {
        return Err("The Ollama download redirected to an unexpected URL.".to_owned());
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    if !target.expected_content_type(content_type) {
        return Err(format!(
            "The Ollama download returned unexpected content type '{content_type}'."
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length != asset.size || length > MAX_DOWNLOAD_BYTES)
    {
        return Err(
            "The Ollama download size did not match the published release asset.".to_owned(),
        );
    }

    let mut file = File::create(destination)
        .await
        .map_err(|error| format!("Could not create temporary installer file: {error}"))?;
    let mut stream = response.bytes_stream();
    let mut downloaded = 0_u64;
    let mut hasher = Sha256::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| format!("Ollama download failed: {error}"))?;
        downloaded = downloaded
            .checked_add(chunk.len() as u64)
            .filter(|length| *length <= MAX_DOWNLOAD_BYTES)
            .ok_or_else(|| "The Ollama download exceeded the size limit.".to_owned())?;
        hasher.update(&chunk);
        file.write_all(&chunk)
            .await
            .map_err(|error| format!("Could not save the Ollama download: {error}"))?;
    }
    file.flush()
        .await
        .map_err(|error| format!("Could not finish saving Ollama: {error}"))?;
    if downloaded != asset.size {
        return Err(format!(
            "The Ollama download was incomplete (received {downloaded} of {} bytes).",
            asset.size
        ));
    }
    if let Some(expected_digest) = &asset.digest {
        let actual_digest = format!("{:x}", hasher.finalize());
        if expected_digest != &format!("sha256:{actual_digest}") {
            return Err("The Ollama download failed its published SHA-256 check.".to_owned());
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

struct TemporaryDirectory(PathBuf);

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn latest_release_asset(
    client: &Client,
    target: InstallerTarget,
) -> Result<(String, ReleaseAsset), String> {
    let release = client
        .get(RELEASE_API)
        .header(ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .map_err(|error| format!("Could not look up the latest Ollama release: {error}"))?
        .error_for_status()
        .map_err(|error| format!("Could not look up the latest Ollama release: {error}"))?
        .json::<Release>()
        .await
        .map_err(|error| format!("Could not read the latest Ollama release metadata: {error}"))?;
    let asset = release
        .assets
        .iter()
        .find(|asset| asset.name == target.asset_name())
        .ok_or_else(|| {
            format!(
                "The latest Ollama release does not include {}.",
                target.asset_name()
            )
        })?;
    validate_release_asset(target, asset)?;
    Ok((
        release.tag_name,
        ReleaseAsset {
            name: asset.name.clone(),
            size: asset.size,
            digest: asset.digest.clone(),
            browser_download_url: asset.browser_download_url.clone(),
        },
    ))
}

#[tauri::command]
pub async fn install_ollama(app: AppHandle) -> Result<String, String> {
    let target = installer_target(env::consts::OS, env::consts::ARCH)?;
    let client = Client::builder()
        .user_agent("Legion desktop app")
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(3 * 60 * 60))
        .build()
        .map_err(|error| format!("Could not configure secure Ollama downloads: {error}"))?;
    let (release_tag, asset) = latest_release_asset(&client, target).await?;
    let temp_path = env::temp_dir().join(format!(
        "legion-ollama-install-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos())
    ));
    fs::create_dir(&temp_path)
        .await
        .map_err(|error| format!("Could not create a temporary install directory: {error}"))?;
    let _temporary_directory = TemporaryDirectory(temp_path.clone());
    let installer_path = temp_path.join(&asset.name);
    let checksum_verified =
        download_release_asset(&client, target, &asset, &installer_path).await?;

    #[cfg(target_os = "windows")]
    install_windows(&app, &installer_path).await?;
    #[cfg(target_os = "macos")]
    install_macos(&app, &installer_path, &temp_path).await?;
    #[cfg(target_os = "linux")]
    install_linux(&app, &installer_path, &temp_path).await?;
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    return Err(
        "Automatic Ollama installation is not supported on this operating system.".to_owned(),
    );

    Ok(if checksum_verified {
        format!(
            "Ollama {} was installed. Its downloaded release asset passed the published SHA-256 check.",
            release_tag
        )
    } else {
        format!(
            "Ollama {} was installed over HTTPS and passed content-type and size checks. The release did not publish a SHA-256 digest, so the download could not be checksum-verified.",
            release_tag
        )
    })
}

#[cfg(target_os = "windows")]
async fn install_windows(_app: &AppHandle, installer_path: &Path) -> Result<(), String> {
    let status = Command::new(installer_path)
        .status()
        .await
        .map_err(|error| format!("Could not launch the Ollama installer: {error}"))?;
    if !status.success() {
        return Err(format!(
            "The Ollama installer exited with status {}.",
            status.code().map_or_else(
                || "without an exit code".to_owned(),
                |code| code.to_string()
            )
        ));
    }

    let local_app_data = env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .ok_or_else(|| "Could not locate the Windows local application directory.".to_owned())?;
    let executable = local_app_data
        .join("Programs")
        .join("Ollama")
        .join("ollama.exe");
    start_server(&executable).await
}

#[cfg(target_os = "macos")]
async fn install_macos(
    app: &AppHandle,
    installer_path: &Path,
    temp_path: &Path,
) -> Result<(), String> {
    let extraction_path = temp_path.join("unpacked");
    fs::create_dir(&extraction_path)
        .await
        .map_err(|error| format!("Could not prepare the Ollama app archive: {error}"))?;
    run_command(
        Command::new("ditto")
            .arg("-x")
            .arg("-k")
            .arg(installer_path)
            .arg(&extraction_path),
        "Could not extract the Ollama app archive",
    )
    .await?;
    let extracted_app = extraction_path.join("Ollama.app");
    if !extracted_app.is_dir() {
        return Err("The official Ollama archive did not contain Ollama.app.".to_owned());
    }

    let applications = app
        .path()
        .home_dir()
        .map_err(|error| format!("Could not locate the user's home directory: {error}"))?
        .join("Applications");
    fs::create_dir_all(&applications)
        .await
        .map_err(|error| format!("Could not create the user's Applications folder: {error}"))?;
    let installed_app = applications.join("Ollama.app");
    let staged_app = applications.join(format!(".Ollama.app.installing-{}", std::process::id()));
    if staged_app.exists() {
        fs::remove_dir_all(&staged_app)
            .await
            .map_err(|error| format!("Could not clean up a previous Ollama install: {error}"))?;
    }
    let _staged_app_directory = TemporaryDirectory(staged_app.clone());
    run_command(
        Command::new("ditto").arg(&extracted_app).arg(&staged_app),
        "Could not copy Ollama into the user's Applications folder",
    )
    .await?;
    replace_directory(
        &staged_app,
        &installed_app,
        &applications.join(".Ollama.app.previous"),
    )
    .await?;
    run_command(
        Command::new("open")
            .arg("-a")
            .arg(&installed_app)
            .arg("--args")
            .arg("hidden"),
        "Could not start the Ollama app",
    )
    .await
}

#[cfg(target_os = "linux")]
async fn install_linux(
    app: &AppHandle,
    installer_path: &Path,
    temp_path: &Path,
) -> Result<(), String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Could not locate Legion's app-data directory: {error}"))?;
    fs::create_dir_all(&app_data_dir)
        .await
        .map_err(|error| format!("Could not prepare Legion's app-data directory: {error}"))?;
    let staging_path = app_data_dir.join(format!(
        "ollama-runtime-staging-{}",
        temp_path.file_name().unwrap_or_default().to_string_lossy()
    ));
    fs::create_dir(&staging_path)
        .await
        .map_err(|error| format!("Could not prepare the Ollama archive: {error}"))?;
    let _staging_directory = TemporaryDirectory(staging_path.clone());
    let archive = File::open(installer_path)
        .await
        .map_err(|error| format!("Could not open the Ollama archive: {error}"))?;
    let decoder = zstd::stream::read::Decoder::new(archive.into_std().await)
        .map_err(|error| format!("Could not decompress the Ollama archive: {error}"))?;
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(&staging_path)
        .map_err(|error| format!("Could not extract the Ollama archive: {error}"))?;
    let extracted_binary = staging_path.join("ollama");
    if !extracted_binary.is_file() {
        return Err("The official Ollama archive did not contain its expected binary.".to_owned());
    }

    let install_path = app_data_dir.join("ollama-runtime");
    replace_directory(
        &staging_path,
        &install_path,
        &app_data_dir.join("ollama-runtime.previous"),
    )
    .await?;
    start_server(&install_path.join("ollama")).await
}

#[cfg(any(target_os = "macos", target_os = "linux", test))]
async fn replace_directory(staged: &Path, installed: &Path, backup: &Path) -> Result<(), String> {
    if backup.exists() {
        fs::remove_dir_all(backup)
            .await
            .map_err(|error| format!("Could not remove the previous Ollama backup: {error}"))?;
    }
    let had_install = installed.exists();
    if had_install {
        fs::rename(installed, backup)
            .await
            .map_err(|error| format!("Could not move the existing Ollama install: {error}"))?;
    }
    if let Err(error) = fs::rename(staged, installed).await {
        if had_install {
            fs::rename(backup, installed).await.map_err(|restore_error| {
                format!(
                    "Could not install Ollama: {error}; restoring the previous install also failed: {restore_error}"
                )
            })?;
        }
        return Err(format!("Could not install Ollama: {error}"));
    }
    if had_install {
        fs::remove_dir_all(backup).await.map_err(|error| {
            format!("Ollama was installed but the old copy could not be removed: {error}")
        })?;
    }
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
async fn run_command(command: &mut Command, action: &str) -> Result<(), String> {
    let status = command
        .status()
        .await
        .map_err(|error| format!("{action}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{action}: process exited with status {}.",
            status.code().map_or_else(
                || "without an exit code".to_owned(),
                |code| code.to_string()
            )
        ))
    }
}

async fn start_server(executable: &Path) -> Result<(), String> {
    let mut command = Command::new(executable);
    command.arg("serve");
    command.spawn().map(|_| ()).map_err(|error| {
        format!("Ollama was installed, but its server could not be started: {error}")
    })
}

#[cfg(test)]
mod tests {
    use std::fs as std_fs;

    use super::{
        installer_target, replace_directory, validate_release_asset, InstallerTarget, ReleaseAsset,
    };

    #[test]
    fn selects_official_release_asset_for_supported_platforms() {
        assert_eq!(
            installer_target("windows", "x86_64"),
            Ok(InstallerTarget::WindowsX64)
        );
        assert_eq!(
            installer_target("macos", "aarch64"),
            Ok(InstallerTarget::Mac)
        );
        assert_eq!(
            installer_target("linux", "x86_64"),
            Ok(InstallerTarget::LinuxX64)
        );
        assert_eq!(
            installer_target("linux", "aarch64"),
            Ok(InstallerTarget::LinuxArm64)
        );
        assert_eq!(InstallerTarget::WindowsX64.asset_name(), "OllamaSetup.exe");
        assert_eq!(InstallerTarget::Mac.asset_name(), "Ollama-darwin.zip");
        assert_eq!(
            InstallerTarget::LinuxX64.asset_name(),
            "ollama-linux-amd64.tar.zst"
        );
        assert_eq!(
            InstallerTarget::LinuxArm64.asset_name(),
            "ollama-linux-arm64.tar.zst"
        );
        assert!(InstallerTarget::Mac.expected_content_type("application/zip"));
        assert!(InstallerTarget::LinuxX64.expected_content_type("application/zstd"));
        assert!(!InstallerTarget::WindowsX64.expected_content_type("text/html"));
    }

    #[test]
    fn rejects_unsupported_platforms_and_architectures() {
        assert!(installer_target("windows", "aarch64").is_err());
        assert!(installer_target("linux", "riscv64").is_err());
        assert!(installer_target("freebsd", "x86_64").is_err());
    }

    #[test]
    fn validates_asset_name_size_digest_and_https_url() {
        let valid = ReleaseAsset {
            name: "OllamaSetup.exe".to_owned(),
            size: 150 * 1024 * 1024,
            digest: Some(format!("sha256:{}", "a".repeat(64))),
            browser_download_url:
                "https://github.com/ollama/ollama/releases/download/v1/OllamaSetup.exe".to_owned(),
        };
        assert_eq!(
            validate_release_asset(InstallerTarget::WindowsX64, &valid)
                .expect("official asset is accepted")
                .scheme(),
            "https"
        );

        let mut invalid = valid;
        invalid.browser_download_url =
            "http://github.com/ollama/ollama/releases/download/v1/OllamaSetup.exe".to_owned();
        assert!(validate_release_asset(InstallerTarget::WindowsX64, &invalid).is_err());
        invalid.browser_download_url = "https://example.com/OllamaSetup.exe".to_owned();
        assert!(validate_release_asset(InstallerTarget::WindowsX64, &invalid).is_err());
        invalid.browser_download_url =
            "https://github.com/ollama/ollama/releases/download/v1/OllamaSetup.exe".to_owned();
        invalid.digest = Some("sha256:bad".to_owned());
        assert!(validate_release_asset(InstallerTarget::WindowsX64, &invalid).is_err());

        invalid.digest = None;
        assert!(validate_release_asset(InstallerTarget::WindowsX64, &invalid).is_ok());
        invalid.size = 1024;
        assert!(validate_release_asset(InstallerTarget::WindowsX64, &invalid).is_err());

        invalid.size = 150 * 1024 * 1024;
        invalid.name = "unexpected.exe".to_owned();
        assert!(validate_release_asset(InstallerTarget::WindowsX64, &invalid).is_err());
    }

    #[tokio::test]
    async fn replacing_runtime_restores_the_previous_install_on_swap_failure() {
        let root = std::env::temp_dir().join(format!(
            "legion-installer-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock is after unix epoch")
                .as_nanos()
        ));
        std_fs::create_dir_all(&root).expect("test directory is created");
        let installed = root.join("installed");
        let backup = root.join("backup");
        let staged = root.join("staged");
        std_fs::create_dir(&installed).expect("old install directory is created");
        std_fs::write(installed.join("version"), "old").expect("old version is written");

        let error = replace_directory(&staged, &installed, &backup)
            .await
            .expect_err("missing staged directory fails");
        assert!(error.contains("Could not install Ollama"));
        assert_eq!(
            std_fs::read_to_string(installed.join("version")).expect("old install is restored"),
            "old"
        );
        assert!(!backup.exists());

        std_fs::create_dir(&staged).expect("staged install directory is created");
        std_fs::write(staged.join("version"), "new").expect("new version is written");
        replace_directory(&staged, &installed, &backup)
            .await
            .expect("new install replaces old install");
        assert_eq!(
            std_fs::read_to_string(installed.join("version")).expect("new install is present"),
            "new"
        );
        std_fs::remove_dir_all(&root).expect("test directory is cleaned up");
    }
}
