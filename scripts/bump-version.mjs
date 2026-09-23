import { readFileSync, writeFileSync } from "node:fs";

const [version] = process.argv.slice(2);
if (
  !version ||
  !/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(
    version,
  )
) {
  throw new Error("Usage: npm run version:bump -- <semver-version>");
}

const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
const packageLock = JSON.parse(readFileSync("package-lock.json", "utf8"));
const tauriConfig = JSON.parse(
  readFileSync("src-tauri/tauri.conf.json", "utf8"),
);
const cargoManifestPath = "src-tauri/Cargo.toml";
const cargoManifest = readFileSync(cargoManifestPath, "utf8");
const cargoVersionPattern = /(\[package\][\s\S]*?\nversion\s*=\s*")[^"]+(")/;
const cargoLockPath = "src-tauri/Cargo.lock";
const cargoLock = readFileSync(cargoLockPath, "utf8");
const cargoLockPackagePattern =
  /(\[\[package\]\]\r?\n(?:(?!\[\[package\]\])[\s\S])*?name = "legion"\r?\nversion = ")[^"]+(")/;

if (!cargoVersionPattern.test(cargoManifest)) {
  throw new Error(
    "Unable to find the package version in src-tauri/Cargo.toml.",
  );
}
if (!cargoLockPackagePattern.test(cargoLock)) {
  throw new Error("Unable to find the legion package in src-tauri/Cargo.lock.");
}
if (!packageLock.packages?.[""]) {
  throw new Error("Unable to find the root package in package-lock.json.");
}

const updatedCargoManifest = cargoManifest.replace(
  cargoVersionPattern,
  (_match, prefix, suffix) => `${prefix}${version}${suffix}`,
);
const updatedCargoLock = cargoLock.replace(
  cargoLockPackagePattern,
  (_match, prefix, suffix) => `${prefix}${version}${suffix}`,
);
packageJson.version = version;
packageLock.version = version;
packageLock.packages[""].version = version;
tauriConfig.version = version;

writeFileSync("package.json", `${JSON.stringify(packageJson, null, 2)}\n`);
writeFileSync("package-lock.json", `${JSON.stringify(packageLock, null, 2)}\n`);
writeFileSync(
  "src-tauri/tauri.conf.json",
  `${JSON.stringify(tauriConfig, null, 2)}\n`,
);
writeFileSync(cargoManifestPath, updatedCargoManifest);
writeFileSync(cargoLockPath, updatedCargoLock);

console.log(`Updated application version to ${version}.`);
