import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const SOURCE_DIR = path.join(ROOT, "assets", "brand");
const OUTPUT_DIR = path.join(
  ROOT,
  "custom_components",
  "learning_thermostat",
  "brand",
);

export const EXPORT_JOBS = Object.freeze([
  ["icon-light.svg", "icon.png", 256, 256],
  ["icon-light.svg", "icon@2x.png", 512, 512],
  ["icon-dark.svg", "dark_icon.png", 256, 256],
  ["icon-dark.svg", "dark_icon@2x.png", 512, 512],
  ["logo-light.svg", "logo.png", 512, 160],
  ["logo-light.svg", "logo@2x.png", 1024, 320],
  ["logo-dark.svg", "dark_logo.png", 512, 160],
  ["logo-dark.svg", "dark_logo@2x.png", 1024, 320],
]);

export function exportJobs() {
  return EXPORT_JOBS.map(([sourceName, outputName, width, height]) => ({
    sourceName,
    outputName,
    width,
    height,
    sourcePath: path.join(SOURCE_DIR, sourceName),
    outputPath: path.join(OUTPUT_DIR, outputName),
  }));
}

export async function renderJob(job) {
  const sourceBuffer = await readFile(job.sourcePath);
  await mkdir(path.dirname(job.outputPath), { recursive: true });
  await sharp(sourceBuffer, { density: 288 })
    .resize(job.width, job.height, { fit: "fill" })
    .png()
    .toFile(job.outputPath);
}

export async function main() {
  for (const job of exportJobs()) {
    await renderJob(job);
    console.log(`exported ${job.sourceName} -> ${job.outputName}`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
