# alainfidahoussen.github.io

Source for my personal blog, built with [Hugo](https://gohugo.io/) and the
[Congo](https://github.com/jpanther/congo) theme, deployed to GitHub Pages via GitHub Actions.

## Local development

```
brew install hugo
git submodule update --init --recursive
hugo server -D
```

Site will be available at http://localhost:1313/

## Deployment

Pushing to `main` triggers `.github/workflows/hugo.yaml`, which builds the site and publishes it
via GitHub Pages (Actions deployment source).
