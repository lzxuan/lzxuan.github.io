import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  // Must match public/CNAME so canonical + Open Graph URLs resolve.
  site: 'https://zhixuanlim.com',
  build: {
    inlineStylesheets: 'auto'
  }
});
