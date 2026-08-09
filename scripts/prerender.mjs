#!/usr/bin/env node
/**
 * Build-time prerendering script.
 *
 * Runs after `vite build` to generate static HTML for every route.
 * Crawlers and AI-powered search see real content instead of an empty <div id="app">.
 *
 * Usage:  node scripts/prerender.mjs          (expects dist/ to exist)
 */

import puppeteer from 'puppeteer';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const PORT = 45678;
const SITE_URL = 'https://victor-retamal.com';

// ---------------------------------------------------------------------------
// Route discovery
// ---------------------------------------------------------------------------

/**
 * Posts, in the order they appear in the YAML (newest first), with the date
 * each was published so the sitemap can carry a truthful lastmod.
 */
function getPosts() {
  const yaml = fs.readFileSync(path.join(ROOT, 'src/data/blog-posts.yaml'), 'utf-8');
  const slugs = [...yaml.matchAll(/slug:\s*"([^"]+)"/g)].map(m => m[1]);
  const dates = [...yaml.matchAll(/date:\s*"([^"]+)"/g)].map(m => m[1]);

  return slugs.map((slug, index) => ({ slug, date: dates[index] || null }));
}

function getRoutes() {
  // /services is intentionally absent: the page still resolves client-side,
  // but it is unlisted, so it stays out of the pre-render and the sitemap.
  return [
    '/',
    '/blog',
    ...getPosts().map(post => `/blog/${post.slug}`),
  ];
}

// ---------------------------------------------------------------------------
// Minimal static file server with SPA fallback
// ---------------------------------------------------------------------------

function createServer(fallbackHtml) {
  const MIME = {
    '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.css': 'text/css', '.json': 'application/json',
    '.yaml': 'text/yaml', '.yml': 'text/yaml',
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.webp': 'image/webp',
    '.woff': 'font/woff', '.woff2': 'font/woff2', '.ttf': 'font/ttf',
    '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
    '.tbin': 'application/octet-stream', '.map': 'application/json',
  };

  return http.createServer((req, res) => {
    const url = new URL(req.url, `http://localhost:${PORT}`);
    const filePath = path.join(DIST, decodeURIComponent(url.pathname));

    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const ext = path.extname(filePath);
      res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
      fs.createReadStream(filePath).pipe(res);
    } else if (fs.existsSync(path.join(filePath, 'index.html'))) {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      fs.createReadStream(path.join(filePath, 'index.html')).pipe(res);
    } else {
      // SPA fallback. This must be the pristine shell, not whatever is
      // currently on disk: by the time posts are rendered, dist/index.html
      // is the finished home page, and serving that would leak the home
      // page's head - structured data included - into every post.
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(fallbackHtml);
    }
  });
}

// ---------------------------------------------------------------------------
// Sitemap generation
// ---------------------------------------------------------------------------

function generateSitemap(routes) {
  const today = new Date().toISOString().split('T')[0];
  const posts = getPosts();

  // A post's lastmod is the day it was published, not the day we happened to
  // build. The listing pages move whenever the newest post does.
  const postDates = new Map(posts.map(post => [`/blog/${post.slug}`, post.date]));
  const newestPost = posts.map(post => post.date).filter(Boolean).sort().pop();

  const lastmodFor = route => {
    if (route === '/' || route === '/blog') return newestPost || today;
    return postDates.get(route) || today;
  };

  const entries = routes.map(route => {
    const priority =
      route === '/'     ? '1.0' :
      route === '/blog' ? '0.9' : '0.7';
    const changefreq = route === '/' ? 'weekly' : 'monthly';

    return [
      '  <url>',
      `    <loc>${SITE_URL}${route}</loc>`,
      `    <lastmod>${lastmodFor(route)}</lastmod>`,
      `    <changefreq>${changefreq}</changefreq>`,
      `    <priority>${priority}</priority>`,
      '  </url>',
    ].join('\n');
  }).join('\n');

  const sitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    entries,
    '</urlset>',
    '',
  ].join('\n');

  fs.writeFileSync(path.join(DIST, 'sitemap.xml'), sitemap);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function prerender() {
  if (!fs.existsSync(DIST)) {
    console.error('Error: dist/ not found. Run "vite build" first.');
    process.exit(1);
  }

  const routes = getRoutes();
  console.log(`\nPre-rendering ${routes.length} routes…\n`);

  // Start local server
  // The shell as vite built it, before any route overwrites index.html
  const fallbackHtml = fs.readFileSync(path.join(DIST, 'index.html'), 'utf-8');

  const server = createServer(fallbackHtml);
  await new Promise(resolve => server.listen(PORT, resolve));

  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-gpu',
      '--disable-dev-shm-usage',
    ],
  });

  let ok = 0;

  for (const route of routes) {
    const page = await browser.newPage();

    // Block heavy resources that crawlers don't need
    await page.setRequestInterception(true);
    page.on('request', req => {
      if (['image', 'font', 'media'].includes(req.resourceType())) {
        req.abort();
      } else {
        req.continue();
      }
    });

    try {
      process.stdout.write(`  ${route} `);
      await page.goto(`http://localhost:${PORT}${route}`, {
        waitUntil: 'networkidle0',
        timeout: 30000,
      });

      // Wait until #app has meaningful content
      await page.waitForFunction(
        () => {
          const app = document.getElementById('app');
          return app && app.innerHTML.trim().length > 100;
        },
        { timeout: 15000 },
      );

      // Let async rendering settle
      await new Promise(r => setTimeout(r, 1500));

      const html = await page.content();

      // Write to correct directory structure
      const outPath = route === '/'
        ? path.join(DIST, 'index.html')
        : path.join(DIST, route, 'index.html');

      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      fs.writeFileSync(outPath, html);
      console.log(`-> ${path.relative(DIST, outPath)}`);
      ok++;
    } catch (err) {
      console.log(`FAILED: ${err.message}`);
      // Write SPA fallback so the page still works client-side
      const outPath = route === '/'
        ? path.join(DIST, 'index.html')
        : path.join(DIST, route, 'index.html');
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      fs.writeFileSync(outPath, fallbackHtml);
    } finally {
      await page.close();
    }
  }

  await browser.close();
  server.close();

  // Regenerate sitemap with all routes
  generateSitemap(routes);
  console.log('  Generated sitemap.xml');

  console.log(`\nDone: ${ok}/${routes.length} pages pre-rendered.\n`);

  if (ok === 0) {
    console.error('All pages failed to pre-render!');
    process.exit(1);
  }
}

prerender().catch(err => {
  console.error('Pre-rendering failed:', err);
  process.exit(1);
});
