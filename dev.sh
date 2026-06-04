#!/bin/bash
# Dygen dev helper. The app builds via Dygen.xcodeproj (consumes the local
# WindowKit/DygenCore Swift packages); package logic tests run via SwiftPM.
#
# Usage: ./dev.sh [build|run|test|clean]
set -euo pipefail
cd "$(dirname "$0")"

PROJECT=Dygen.xcodeproj
SCHEME=Dygen
DD=build/dd
APP="$DD/Build/Products/Debug/$SCHEME.app"

build() {
    xcodebuild -project "$PROJECT" -scheme "$SCHEME" -configuration Debug \
        -destination 'platform=macOS,arch=arm64' -derivedDataPath "$DD" build \
        2>&1 | grep -E "error:|warning:|BUILD (SUCCEEDED|FAILED)" || true
}

case "${1:-run}" in
    build) build ;;
    run)   build && open "$APP" ;;
    test)  swift test ;;          # WindowKit/DygenCore logic tests
    clean) rm -rf "$DD" .build ;;
    *) echo "Usage: $0 [build|run|test|clean]"; exit 1 ;;
esac
