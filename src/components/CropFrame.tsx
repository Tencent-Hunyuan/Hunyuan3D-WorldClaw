import type { CSSProperties, ReactNode } from "react";
import type { CropRect } from "../data/content";

const FIGURE_WIDTH = 1524;
const FIGURE_HEIGHT = 1800;

interface CropFrameProps {
  src: string;
  alt: string;
  rect: CropRect;
  className?: string;
  eager?: boolean;
  children?: ReactNode;
}

type CropStyle = CSSProperties & {
  "--crop-ar": number;
  "--crop-x": number;
  "--crop-y": number;
  "--crop-w": number;
  "--crop-h": number;
};

/**
 * Shows one measured region of a composite paper figure without needing the
 * individual crops as separate assets.
 */
export function CropFrame({
  src,
  alt,
  rect,
  className,
  eager = false,
  children,
}: CropFrameProps) {
  const style: CropStyle = {
    "--crop-ar": (rect.w * FIGURE_WIDTH) / (rect.h * FIGURE_HEIGHT),
    "--crop-x": rect.x,
    "--crop-y": rect.y,
    "--crop-w": rect.w,
    "--crop-h": rect.h,
  };

  return (
    <span className={`crop-frame ${className ?? ""}`.trim()} style={style}>
      <img
        src={src}
        alt={alt}
        width={Math.round(rect.w * FIGURE_WIDTH)}
        height={Math.round(rect.h * FIGURE_HEIGHT)}
        loading={eager ? "eager" : "lazy"}
        decoding="async"
      />
      {children}
    </span>
  );
}
