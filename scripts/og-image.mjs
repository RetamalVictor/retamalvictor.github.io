#!/usr/bin/env node
/**
 * Build-time Open Graph card.
 *
 * The site advertises /images/og-image.png in its meta tags, so every share
 * on a social platform or chat app renders it. Rather than keeping a binary
 * in the repo that drifts from the design, the card is rendered from the same
 * palette the site uses.
 *
 * Usage:  node scripts/og-image.mjs        (expects dist/ to exist)
 */

import puppeteer from 'puppeteer';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT, 'dist', 'images');
const OUT_FILE = path.join(OUT_DIR, 'og-image.png');

const WIDTH = 1200;
const HEIGHT = 630;

const CARD = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    width: ${WIDTH}px; height: ${HEIGHT}px;
    display: flex; align-items: center; justify-content: center;
    background-color: #eaf1fb;
    background-image:
      linear-gradient(rgba(11, 87, 208, 0.07) 1px, transparent 1px),
      linear-gradient(90deg, rgba(11, 87, 208, 0.07) 1px, transparent 1px);
    background-size: 40px 40px;
    font-family: Verdana, Geneva, Tahoma, sans-serif;
  }
  .win {
    width: 1000px;
    background: #ffffff;
    border: 1px solid #1e4fa3;
    border-radius: 10px;
    box-shadow: 10px 10px 0 rgba(11, 37, 69, 0.16);
    overflow: hidden;
  }
  .bar {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 20px;
    background: linear-gradient(180deg, #4a90e2 0%, #1a5fc0 50%, #0b46a0 100%);
    color: #fff; font-size: 18px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
  }
  .controls { margin-left: auto; display: flex; gap: 8px; }
  .controls i {
    width: 18px; height: 18px; border-radius: 3px;
    border: 1px solid rgba(255,255,255,0.45); background: rgba(255,255,255,0.18);
  }
  .body { padding: 56px 60px 60px; }
  .badge {
    display: inline-block; padding: 8px 16px; margin-bottom: 28px;
    font-size: 16px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    color: #0b57d0; background: rgba(11, 87, 208, 0.08);
    border: 1px solid rgba(11, 87, 208, 0.35); border-radius: 4px;
  }
  h1 {
    font-family: Inter, 'Segoe UI', system-ui, sans-serif;
    font-size: 76px; font-weight: 900; line-height: 1.05; color: #0b2545;
    letter-spacing: -1.5px; text-shadow: 5px 5px 0 rgba(11, 87, 208, 0.16);
  }
  h1 span { color: #0b57d0; }
  p {
    margin-top: 26px; font-family: Inter, 'Segoe UI', system-ui, sans-serif;
    font-size: 27px; line-height: 1.5; color: #33587e;
  }
  .foot {
    margin-top: 38px; display: flex; align-items: center; gap: 14px;
    font-size: 17px; color: #5b7899; letter-spacing: 1px;
  }
  .mark {
    width: 40px; height: 40px; border-radius: 6px; background: #0b57d0;
    color: #fff; font-size: 15px; font-weight: 900;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 4px 4px 0 rgba(18, 58, 148, 0.5);
  }
</style>
</head>
<body>
  <div class="win">
    <div class="bar">
      <span>victor-retamal.com</span>
      <span class="controls"><i></i><i></i><i></i></span>
    </div>
    <div class="body">
      <div class="badge">Machine Learning &amp; Robotics Engineer</div>
      <h1>I'm <span>Victor</span>,<br>I teach machines to think and move.</h1>
      <p>Sim-to-real robotics &middot; multi-agent RL &middot; computer vision &middot; medical imaging</p>
      <div class="foot">
        <span class="mark">VR</span>
        <span>Simulations on this site run in your browser</span>
      </div>
    </div>
  </div>
</body>
</html>`;

async function generate() {
  if (!fs.existsSync(path.join(ROOT, 'dist'))) {
    console.error('Error: dist/ not found. Run "vite build" first.');
    process.exit(1);
  }

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 });
  await page.setContent(CARD, { waitUntil: 'load' });

  fs.mkdirSync(OUT_DIR, { recursive: true });
  await page.screenshot({ path: OUT_FILE, type: 'png' });

  await browser.close();

  const kb = (fs.statSync(OUT_FILE).size / 1024).toFixed(0);
  console.log(`\nOpen Graph card -> images/og-image.png (${WIDTH}x${HEIGHT}, ${kb} kB)`);
}

generate().catch(err => {
  console.error('OG image generation failed:', err);
  process.exit(1);
});
