#!/bin/bash
# Point this repo's build at the installed Isaac Sim's own Kit SDK.
#
# This repo deliberately does NOT packman-fetch a kit-kernel package (see the comment in
# deps/kit-sdk.packman.xml). Instead _build/<platform>/<config>/kit is a symlink to the real
# Isaac Sim install, so extensions are compiled and linked against the exact same Kit build
# that will load them at runtime.
#
# ./repo.sh build resolves that path during its first step (packman dependency fetch), so the
# symlink has to exist before a build. Re-run this script after any clean (./build.sh -c / -x),
# which deletes _build/ and the symlink along with it.

set -e

KIT_INSTALL="${KIT_INSTALL:-/home/isaac/isaacsim/kit}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINK_DIR="${SCRIPT_DIR}/_build/linux-x86_64/release"

if [[ ! -d "${KIT_INSTALL}" ]]; then
    echo "error: Kit SDK not found at ${KIT_INSTALL}" >&2
    echo "       set KIT_INSTALL=<path to your isaacsim/kit> and re-run" >&2
    exit 1
fi

mkdir -p "${LINK_DIR}"
ln -sfn "${KIT_INSTALL}" "${LINK_DIR}/kit"
echo "linked ${LINK_DIR}/kit -> ${KIT_INSTALL}"
