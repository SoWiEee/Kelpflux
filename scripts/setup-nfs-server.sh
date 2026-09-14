#!/usr/bin/env bash
set -euo pipefail

# Run on the Linux NFS host with root privileges.

NFS_EXPORT_PATH=${NFS_EXPORT_PATH:-/srv/nfs/k8s}

# Comma-separated node IPs/CIDRs; override when the cluster membership changes.
NFS_EXPORT_CLIENTS=${NFS_EXPORT_CLIENTS:-192.168.0.111,192.168.0.104}

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root: sudo bash scripts/setup-nfs-server.sh" >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y nfs-kernel-server

mkdir -p "$NFS_EXPORT_PATH"
chown nobody:nogroup "$NFS_EXPORT_PATH"
chmod 0755 "$NFS_EXPORT_PATH"

IFS=',' read -r -a export_clients <<< "$NFS_EXPORT_CLIENTS"
exports_line="$NFS_EXPORT_PATH"
for client in "${export_clients[@]}"; do
  if [[ ! "$client" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}(/[0-9]{1,2})?$ ]]; then
    echo "Invalid NFS export client: $client" >&2
    exit 1
  fi
  exports_line+=" ${client}(rw,sync,no_subtree_check,root_squash)"
done

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
