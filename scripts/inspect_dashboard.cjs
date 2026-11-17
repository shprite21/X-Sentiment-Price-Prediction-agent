// Local QA only: uses the host's installed Playwright package and Edge/Chrome.
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

(async () => {
  const input = path.resolve(process.argv[2]);
  const output = path.resolve(process.argv[3]);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1700 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(pathToFileURL(input).href);
    await page.waitForFunction(() => [...document.querySelectorAll('.js-plotly-plot')].length === 6 && [...document.querySelectorAll('.js-plotly-plot')].every(p => p._fullLayout));
    await page.screenshot({ path: output, fullPage: true });
    const details = await page.evaluate(() => {
      const plot = document.querySelector('.js-plotly-plot');
      return { title: document.querySelector('h1').textContent, charts: document.querySelectorAll('.js-plotly-plot').length,
               chartWidth: plot.getBoundingClientRect().width,
               viewportWidth: innerWidth, bodyWidth: document.body.scrollWidth };
    });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => Promise.all([...document.querySelectorAll('.js-plotly-plot')].map(p => Plotly.Plots.resize(p))));
    await page.screenshot({ path: output.replace('.png', '-mobile.png'), fullPage: true });
    console.log(JSON.stringify({ ...details, errors }));
    if (errors.length) process.exitCode = 1;
  } finally { await browser.close(); }
})().catch(e => { console.error(e.message); process.exitCode = 1; });
