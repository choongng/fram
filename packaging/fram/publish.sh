#!/usr/bin/env bash
# Build the fram wheel and publish it to a private package index.
#
#   FRAM_INDEX_URL=https://... UV_PUBLISH_TOKEN=... packaging/fram/publish.sh
#
# FRAM_INDEX_URL is the index's upload URL. Credentials come from uv's own
# environment variables (UV_PUBLISH_TOKEN, or UV_PUBLISH_USERNAME and
# UV_PUBLISH_PASSWORD). Publish only from main; see dev-notes/fram-branches.md.
set -euo pipefail
: "${FRAM_INDEX_URL:?set FRAM_INDEX_URL to the index upload URL}"
here=$(cd "$(dirname "$0")" && pwd)
dist=$(mktemp -d)
trap 'rm -rf "$dist"' EXIT
uv build --wheel --project "$here" --out-dir "$dist"
uv publish --publish-url "$FRAM_INDEX_URL" "$dist"/*.whl
