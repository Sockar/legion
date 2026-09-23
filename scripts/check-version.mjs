import { readFileSync } from "node:fs";

const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
const packageLock = JSON.parse(readFileSync("package-lock.json", "utf8"));
const tauriConfig = JSON.parse(
  readFileSync("src-tauri/tauri.conf.json", "utf8"),
);
const cargoManifest = readFileSync("src-tauri/Cargo.toml", "utf8");
const cargoPackage = cargoManifest
  .split(/^\[package\]\s*$/m)[1]
  ?.split(/^\[/m)[0];
const cargoVersion = cargoPackage?.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
const cargoLock = readFileSync("src-tauri/Cargo.lock", "utf8");
const cargoLockPackage = cargoLock
  .split(/^\[\[package\]\]\s*$/m)
  .find((section) => /^name = "legion"$/m.test(section));
const cargoLockVersion = cargoLockPackage?.match(/^version = "([^"]+)"/m)?.[1];

if (!cargoVersion || !cargoLockVersion) {
  throw new Error("Unable to read the Cargo package versions.");
}

const versions = {
  "package.json": packageJson.version,
  "package-lock.json": packageLock.version,
  "package-lock.json root": packageLock.packages?.[""]?.version,
  "src-tauri/Cargo.toml": cargoVersion,
  "src-tauri/Cargo.lock": cargoLockVersion,
  "src-tauri/tauri.conf.json": tauriConfig.version,
};
const uniqueVersions = new Set(Object.values(versions));

if (uniqueVersions.size !== 1) {
  throw new Error(
    `Application versions must match the package.json version (${packageJson.version}): ${JSON.stringify(versions)}`,
  );
}

if (process.env.GITHUB_REF_TYPE === "tag") {
  const expectedTag = `v${packageJson.version}`;
  if (process.env.GITHUB_REF_NAME !== expectedTag) {
    throw new Error(
      `Release tag ${process.env.GITHUB_REF_NAME ?? "(missing)"} does not match ${expectedTag}.`,
    );
  }

  const publicKey = tauriConfig.plugins?.updater?.pubkey;
  if (!publicKey || publicKey === "__TAURI_UPDATER_PUBLIC_KEY__") {
    throw new Error(
      "Configure the Tauri updater public key in src-tauri/tauri.conf.json before releasing.",
    );
  }
  if (!process.env.TAURI_SIGNING_PRIVATE_KEY) {
    throw new Error(
      "Set the matching TAURI_SIGNING_PRIVATE_KEY repository secret before releasing.",
    );
  }
}

console.log(`Application version ${packageJson.version} is consistent.`);
