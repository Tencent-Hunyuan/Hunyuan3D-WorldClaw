interface BrandMarkProps {
  className?: string;
}

/**
 * WorldClaw mark: an isometric world hull enclosing a generated terrain ridge.
 * The hull is drawn as a hairline and the ridge is solid, so the silhouette
 * still reads at favicon sizes where a fully outlined mark would fill in.
 */
export function BrandMark({ className }: BrandMarkProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M12 1.9 20.75 6.95v10.1L12 22.1 3.25 17.05V6.95z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path
        d="M6.35 16.15 10.05 10.2 12.15 13.35 14.85 8.05 17.65 16.15z"
        fill="currentColor"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
    </svg>
  );
}
