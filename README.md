# Social Segregation in American Cities

Interactive visualization of social segregation patterns in major American cities, exploring mobility and communication networks across census tracts.

**MIT Media Lab × NECSI**

## Features

- **Landing page** with full-scroll sections: Overview, How It Works, City Selection, Authors, References
- **Interactive map** (Leaflet) with ~120 census tracts per city, colored by race or income
- **Network visualization**: click any tract to see mobility or mentions connections
- **Direction toggle**: view incoming vs outgoing connections
- **5 cities**: Chicago, Dallas, New York, Detroit, Philadelphia
- **Responsive** design with dark theme

## Setup

```bash
npm install
npm run dev
```

## Adding your own images

Place images in `public/img/` and update the `IMG_BASE` constant in `src/pages/HomePage.jsx`:

```js
const IMG_BASE = "/img";  // instead of the EC2 URL
```

### Required images

| File | Description |
|------|-------------|
| `jose_image.jpg` | Hero background |
| `Mit_medialab_logo.png` | MIT Media Lab logo |
| `Logo2.png` | NECSI logo |
| `image1.png` | Step 1 screenshot |
| `image2.png` | Step 2 screenshot |
| `image3.png` | Step 3 screenshot |
| `Phil.png` | Philadelphia map thumbnail |
| `Detr.png` | Detroit map thumbnail |
| `Dal.png` | Dallas map thumbnail |
| `Chic.png` | Chicago map thumbnail |
| `NY.png` | New York map thumbnail |

## Deploy to GitHub Pages

```bash
# Update vite.config.js base to match your repo name
npm run deploy
```

## Tech Stack

- React 18 + React Router
- Leaflet (dark CARTO tiles)
- Vite
- gh-pages for deployment
# segregation_map
