#!/usr/bin/env bash
set -euo pipefail

# Run on the Linux NFS host with root privileges.

NFS_EXPORT_PATH=${NFS_EXPORT_PATH:-/srv/nfs/k8s}

# Override this for a different node LAN subnet.
NFS_EXPORT_CLIENTS=${NFS_EXPORT_CLIENTS:-192.168.0.0/24}

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root: sudo bash scripts/setup-nfs-server.sh" >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y nfs-kernel-server

mkdir -p "$NFS_EXPORT_PATH"
chown nobody:nogroup "$NFS_EXPORT_PATH"
chmod 0777 "$NFS_EXPORT_PATH"

exports_line="${NFS_EXPORT_PATH} ${NFS_EXPORT_CLIENTS}(rw,sync,no_subtree_check,no_root_squash)"

if ! grep -qE "^\s*${NFS_EXPORT_PATH}\s" /etc/exports; then
  echo "$exports_line" >> /etc/exports
else
  # Replace existing line for this export path (best-effort)
  sed -i -E "s|^\s*${NFS_EXPORT_PATH}\s.*|${exports_line}|g" /etc/exports
fi

exportfs -ra

systemctl enable --now nfs-kernel-server

echo "NFS server ready"
echo "Export path: ${NFS_EXPORT_PATH}"
echo "Allowed clients: ${NFS_EXPORT_CLIENTS}"
echo "Check with: exportfs -v"
