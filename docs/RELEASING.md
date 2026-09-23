# Releasing Legion

## One-time updater signing setup

Tauri updater payload signatures are generated with a self-managed keypair; they
are separate from Windows/macOS operating-system signing certificates.

1. Generate a keypair on a trusted machine:

   ```sh
   npm run tauri signer generate -- -w ~/.tauri/legion.key
   ```

2. Copy the generated public key into `plugins.updater.pubkey` in
   `src-tauri/tauri.conf.json`, replacing `__TAURI_UPDATER_PUBLIC_KEY__`.
3. Add the matching private key contents as the repository Actions secret
   `TAURI_SIGNING_PRIVATE_KEY`. If the key is password-protected, also add
   `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.

Keep the private key and its password out of the repository. Back up the key
securely: existing installations trust this public key, so losing the matching
private key prevents publishing compatible updates. The tag workflow checks
that the public key and private-key secret are configured before building.

## Cutting a release

`package.json` is the canonical application version. Run the version command,
commit the resulting version updates, and tag that commit using the matching
`v`-prefixed version:

```sh
npm run version:bump -- 0.1.1
npm run check:version
git add package.json package-lock.json src-tauri/Cargo.toml src-tauri/Cargo.lock src-tauri/tauri.conf.json
git commit -m "Bump version to 0.1.1"
git tag v0.1.1
git push origin main
git push origin v0.1.1
```

The version command updates the package manifest, npm lockfile, Cargo manifest
and lockfile, and Tauri configuration together. CI rejects mismatched versions
and tags that do not equal `v<package.json version>`.

Pushing the tag starts the release matrix for Windows (NSIS installer), Linux
(DEB and AppImage), and macOS (DMG). Tauri creates updater artifacts and signs
them with the configured updater key. The workflow creates a **published
GitHub Release** (not a draft), uploads the installers, signatures, and
`latest.json`, and generates release notes. Each platform build contributes its
signed update entry to `latest.json`; the app's Settings panel can then check
for updates and prompt to install one. Linux self-updates use the AppImage
artifact; the DEB is provided for manual installation.

## Optional operating-system signing

Updater signatures are free and self-managed, but do not replace OS-level
signing. To enable that later, configure these repository Actions secrets for
`tauri-apps/tauri-action`:

- Windows Authenticode: `WINDOWS_CERTIFICATE` (base64-encoded certificate) and
  `WINDOWS_CERTIFICATE_PASSWORD`.
- macOS signing and notarization: `APPLE_CERTIFICATE`,
  `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`, `APPLE_ID`,
  `APPLE_PASSWORD`, and `APPLE_TEAM_ID`.

Windows code-signing certificates and Apple Developer accounts/notarization
credentials are not provided by this workflow and may require paid services.
