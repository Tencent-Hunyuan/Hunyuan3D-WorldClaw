#!/bin/bash
#
# Turns the raw Blender turntable renders into the web clips the Results
# section plays, plus a poster frame for each one.
#
#   in   <SRC>/<Prefix>__<cycles|instance_3d|normal|depth>.mp4
#   out  public/media/scenes/<scene-id>/<rgb|instance|normal|depth>.mp4
#        public/media/scenes/<scene-id>/<rgb|instance|normal|depth>.webp
#
# The delivered masters are ~11 Mbps 720p37.5 and total about 500 MB, which is
# far more than a project page should push at a reader. This pass lands the
# same 44 clips at roughly 63 MB with no visible loss at the sizes they are
# displayed (a ~300px grid tile and a ~1100px lightbox).
#
# Usage:  SRC=/path/to/renders scripts/encode-clips.sh
# Needs:  ffmpeg (brew install ffmpeg), python3 with Pillow (for WebP posters,
#         since the Homebrew ffmpeg bottle ships without a WebP encoder).
#
set -u

SRC="${SRC:-$HOME/Downloads/video}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/public/media/scenes"
MAXJOBS="${MAXJOBS:-4}"

# scene-id|source prefix.
# Verified by extracting a frame from each master and matching it against
# public/assets/layouts/<scene-id>.webp — the prefixes carry internal names
# that do not line up with the scene ids by themselves.
MAP=(
  "frontier-mosaic|Scene_Village_v1_sky"
  "snowline-village|Snowy_HunYuan_v1"
  "painted-dunes|Desert_v1_yellowsky"
  "island-settlement|SeaLand_v1_bluesky"
  "grand-canyon|RealMount_rescaled_bluesky"
  "azure-archipelago|LargeSeaLand_v0_statue_replaced_adapted_treefix"
  "ember-caldera|Volcanic_HunYuan_v0"
  "desert-frontier|Oasis_HunYuan_v1_sky"
  "frontier-mine|RealRock_noadapt"
  "verdant-valley|FlatMount_v1"
  "snowbound-outpost|Scene-RealSnowy_bluesky"
)

# out channel|source suffix|crf.
# rgb is the one people actually look at, so it keeps more bits; the data
# channels are flat-shaded and survive a higher crf untouched.
CHANNELS=(
  "rgb|cycles|28"
  "instance|instance_3d|30"
  "normal|normal|30"
  "depth|depth|30"
)

command -v ffmpeg >/dev/null || { echo "ffmpeg not found (brew install ffmpeg)"; exit 1; }
[ -d "$SRC" ] || { echo "source dir not found: $SRC"; exit 1; }

encode_one() {
  local in="$1" outbase="$2" crf="$3"

  # -movflags +faststart puts the moov atom first so playback can begin
  # before the file has finished downloading. Without it the browser has to
  # fetch the whole clip before showing a frame.
  if ! ffmpeg -v error -i "$in" \
      -c:v libx264 -profile:v high -pix_fmt yuv420p \
      -crf "$crf" -preset slow -r 30 -g 60 \
      -vf 'scale=960:540:flags=lanczos' \
      -an -movflags +faststart -y "${outbase}.mp4"; then
    echo "FAIL clip $outbase"; return 1
  fi

  # Poster is the encoded clip's own first frame, so the still the visitor
  # sees is exactly the frame the video starts on.
  if ! ffmpeg -v error -i "${outbase}.mp4" -frames:v 1 \
      -vf 'scale=640:360:flags=lanczos' -y "${outbase}-poster.png"; then
    echo "FAIL poster $outbase"; return 1
  fi
  python3 -c "
from PIL import Image
Image.open('${outbase}-poster.png').convert('RGB').save('${outbase}.webp', quality=78, method=6)
" && rm -f "${outbase}-poster.png"

  echo "ok  $(basename "$(dirname "$outbase")")/$(basename "$outbase")"
}

for entry in "${MAP[@]}"; do
  sid="${entry%%|*}"; prefix="${entry#*|}"
  mkdir -p "$DEST/$sid"
  for ch in "${CHANNELS[@]}"; do
    out="${ch%%|*}"; rest="${ch#*|}"; suffix="${rest%%|*}"; crf="${rest##*|}"
    in="$SRC/${prefix}__${suffix}.mp4"
    [ -f "$in" ] || { echo "MISSING $in"; continue; }

    encode_one "$in" "$DEST/$sid/$out" "$crf" &
    # macOS ships bash 3.2, which has no `wait -n`; poll instead.
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 0.5; done
  done
done
wait

echo "=== done ==="
find "$DEST" -name '*.mp4'  | wc -l | xargs echo "clips   :"
find "$DEST" -name '*.webp' | wc -l | xargs echo "posters :"
du -sh "$DEST"
