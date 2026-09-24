use std::{
    fs,
    path::{Component, Path, PathBuf},
};

pub fn resolve_workspace_path(
    workspace: &Path,
    relative_path: &Path,
    require_existing: bool,
) -> Result<PathBuf, String> {
    if relative_path.as_os_str().is_empty()
        || relative_path.is_absolute()
        || relative_path.components().any(|component| {
            matches!(
                component,
                Component::Prefix(_) | Component::RootDir | Component::ParentDir
            )
        })
    {
        return Err("Path must be relative to the workspace and cannot contain '..'".to_owned());
    }

    let root = fs::canonicalize(workspace)
        .map_err(|error| format!("Could not resolve workspace folder: {error}"))?;
    if !root.is_dir() {
        return Err("Workspace path is not a directory".to_owned());
    }

    let candidate = root.join(relative_path);
    let mut existing = candidate.as_path();
    while !existing.exists() && !fs::symlink_metadata(existing).is_ok() {
        existing = existing
            .parent()
            .ok_or_else(|| "Path has no workspace parent".to_owned())?;
    }
    let resolved_existing = ensure_within_workspace(
        &root,
        fs::canonicalize(existing)
            .map_err(|error| format!("Could not resolve path within workspace: {error}"))?,
    )?;

    let resolved = if require_existing {
        fs::canonicalize(&candidate)
            .map_err(|error| format!("Could not resolve path within workspace: {error}"))?
    } else {
        let remaining = candidate
            .strip_prefix(existing)
            .map_err(|_| "Path escapes the workspace folder".to_owned())?;
        resolved_existing.join(remaining)
    };
    ensure_within_workspace(&root, resolved)
}

pub fn resolve_path_candidate(
    workspace: &Path,
    requested_path: &Path,
    require_existing: bool,
) -> Result<(PathBuf, PathBuf), String> {
    let root = fs::canonicalize(workspace)
        .map_err(|error| format!("Could not resolve workspace folder: {error}"))?;
    if !root.is_dir() {
        return Err("Workspace path is not a directory".to_owned());
    }

    let candidate = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        root.join(requested_path)
    };
    let resolved = if require_existing || fs::symlink_metadata(&candidate).is_ok() {
        fs::canonicalize(&candidate)
            .map_err(|error| format!("Could not resolve requested path: {error}"))?
    } else {
        let mut existing = candidate.as_path();
        while fs::symlink_metadata(existing).is_err() {
            existing = existing
                .parent()
                .ok_or_else(|| "Path has no existing parent".to_owned())?;
        }
        let resolved_existing = fs::canonicalize(existing)
            .map_err(|error| format!("Could not resolve requested path parent: {error}"))?;
        let remaining = candidate
            .strip_prefix(existing)
            .map_err(|_| "Could not resolve requested path parent".to_owned())?;
        resolved_existing.join(remaining)
    };
    Ok((root, resolved))
}

pub fn relative_workspace_path(
    workspace_root: &Path,
    absolute_path: &Path,
) -> Result<PathBuf, String> {
    let root = fs::canonicalize(workspace_root)
        .map_err(|error| format!("Could not resolve workspace folder: {error}"))?;
    let absolute_path = ensure_within_workspace(
        &root,
        fs::canonicalize(absolute_path)
            .map_err(|error| format!("Could not resolve workspace entry: {error}"))?,
    )?;
    let relative = absolute_path
        .strip_prefix(&root)
        .map_err(|_| "Path escapes the workspace folder".to_owned())?;
    Ok(relative.to_path_buf())
}

fn ensure_within_workspace(root: &Path, path: PathBuf) -> Result<PathBuf, String> {
    if path.starts_with(root) {
        Ok(path)
    } else {
        Err("Path escapes the workspace folder".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::{relative_workspace_path, resolve_path_candidate, resolve_workspace_path};
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temporary_directory() -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("legion-paths-{id}"));
        fs::create_dir_all(&path).expect("temporary directory is created");
        path
    }

    #[test]
    fn accepts_workspace_relative_paths_and_workspace_root() {
        let root = temporary_directory();
        fs::write(root.join("file.txt"), "content").expect("file is created");

        assert_eq!(
            resolve_workspace_path(&root, Path::new("file.txt"), true)
                .expect("workspace file resolves"),
            fs::canonicalize(root.join("file.txt")).expect("file canonicalizes")
        );
        assert_eq!(
            resolve_workspace_path(&root, Path::new("."), true).expect("workspace root resolves"),
            fs::canonicalize(&root).expect("root canonicalizes")
        );
        fs::remove_dir_all(root).expect("temporary directory is removed");
    }

    #[test]
    fn returns_relative_paths_only_for_workspace_entries() {
        let parent = temporary_directory();
        let root = parent.join("workspace");
        let outside = parent.join("outside.txt");
        fs::create_dir(&root).expect("workspace is created");
        fs::write(root.join("inside.txt"), "inside").expect("workspace file is created");
        fs::write(&outside, "outside").expect("outside file is created");

        assert_eq!(
            relative_workspace_path(&root, &root.join("inside.txt"))
                .expect("workspace entry resolves"),
            PathBuf::from("inside.txt")
        );
        assert!(relative_workspace_path(&root, &outside).is_err());
        fs::remove_dir_all(parent).expect("temporary directory is removed");
    }

    #[test]
    fn rejects_parent_and_absolute_paths_for_existing_and_new_files() {
        let root = temporary_directory();
        let outside = root.parent().expect("temporary directory has a parent");
        for path in ["../outside.txt", "nested/../../outside.txt"] {
            assert!(resolve_workspace_path(&root, Path::new(path), true).is_err());
            assert!(resolve_workspace_path(&root, Path::new(path), false).is_err());
        }
        assert!(resolve_workspace_path(&root, outside, true).is_err());
        fs::remove_dir_all(root).expect("temporary directory is removed");
    }

    #[test]
    fn resolves_outside_candidates_for_later_authorization() {
        let parent = temporary_directory();
        let workspace = parent.join("workspace");
        let outside = parent.join("outside.txt");
        fs::create_dir(&workspace).expect("workspace is created");
        fs::write(&outside, "outside").expect("outside file is created");

        assert_eq!(
            resolve_path_candidate(&workspace, Path::new("../outside.txt"), true)
                .expect("outside path can be identified")
                .1,
            fs::canonicalize(&outside).expect("outside file canonicalizes")
        );
        assert_eq!(
            resolve_path_candidate(&workspace, Path::new("../new.txt"), false)
                .expect("new outside path can be identified")
                .1,
            parent.join("new.txt")
        );

        fs::remove_dir_all(parent).expect("temporary directory is removed");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinks_that_resolve_outside_the_workspace() {
        use std::os::unix::fs::symlink;

        let parent = temporary_directory();
        let root = parent.join("workspace");
        let outside = parent.join("outside");
        fs::create_dir(&root).expect("workspace is created");
        fs::create_dir(&outside).expect("outside directory is created");
        fs::write(outside.join("secret.txt"), "secret").expect("outside file is created");
        symlink(&outside, root.join("linked")).expect("directory symlink is created");

        assert!(resolve_workspace_path(&root, Path::new("linked/secret.txt"), true).is_err());
        assert!(resolve_workspace_path(&root, Path::new("linked/new.txt"), false).is_err());
        fs::remove_dir_all(parent).expect("temporary directories are removed");
    }

    #[cfg(windows)]
    #[test]
    fn rejects_symlinks_that_resolve_outside_the_workspace() {
        use std::os::windows::fs::symlink_dir;

        let parent = temporary_directory();
        let root = parent.join("workspace");
        let outside = parent.join("outside");
        fs::create_dir(&root).expect("workspace is created");
        fs::create_dir(&outside).expect("outside directory is created");
        fs::write(outside.join("secret.txt"), "secret").expect("outside file is created");
        if let Err(error) = symlink_dir(&outside, root.join("linked")) {
            if error.raw_os_error() == Some(1314) {
                fs::remove_dir_all(parent).expect("temporary directories are removed");
                return;
            }
            panic!("directory symlink is created: {error}");
        }

        assert!(resolve_workspace_path(&root, Path::new("linked/secret.txt"), true).is_err());
        assert!(resolve_workspace_path(&root, Path::new("linked/new.txt"), false).is_err());
        fs::remove_dir_all(parent).expect("temporary directories are removed");
    }
}
