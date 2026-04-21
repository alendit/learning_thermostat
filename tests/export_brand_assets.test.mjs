import test from "node:test";
import assert from "node:assert/strict";

import { exportJobs } from "../scripts/export_brand_assets.mjs";

test("export jobs cover expected runtime outputs", () => {
  const jobs = exportJobs();
  assert.deepEqual(
    jobs.map(({ outputName, width, height }) => [outputName, width, height]),
    [
      ["icon.png", 256, 256],
      ["icon@2x.png", 512, 512],
      ["dark_icon.png", 256, 256],
      ["dark_icon@2x.png", 512, 512],
      ["logo.png", 512, 160],
      ["logo@2x.png", 1024, 320],
      ["dark_logo.png", 512, 160],
      ["dark_logo@2x.png", 1024, 320],
    ],
  );
});

test("export jobs read from svg source assets", () => {
  const jobs = exportJobs();
  assert.deepEqual(
    new Set(jobs.map(({ sourceName }) => sourceName)),
    new Set([
      "icon-light.svg",
      "icon-dark.svg",
      "logo-light.svg",
      "logo-dark.svg",
    ]),
  );
});
