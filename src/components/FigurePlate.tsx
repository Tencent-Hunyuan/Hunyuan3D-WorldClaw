import {
  ArrowsOut,
  MagnifyingGlassMinus,
  MagnifyingGlassPlus,
  X,
} from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { type ReactNode, useEffect, useRef, useState } from "react";

interface FigurePlateProps {
  className?: string;
  src: string;
  alt: string;
  width: number;
  height: number;
  label: string;
  caption: ReactNode;
  /** Plain-text caption for the enlarged view's accessible name. */
  captionText: string;
}

/**
 * Paper figures are dense enough that they are unreadable at column width, so
 * every plate doubles as a button that opens the full-resolution version.
 */
export function FigurePlate({
  className,
  src,
  alt,
  width,
  height,
  label,
  caption,
  captionText,
}: FigurePlateProps) {
  const [open, setOpen] = useState(false);
  const [actualSize, setActualSize] = useState(false);
  const reduceMotion = useReducedMotion();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  const close = () => {
    setOpen(false);
    setActualSize(false);
  };

  useEffect(() => {
    if (!open) return;

    const previousOverflow = document.body.style.overflow;
    const trigger = triggerRef.current;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };

    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", onKeyDown);
    closeRef.current?.focus();

    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
      trigger?.focus();
    };
  }, [open]);

  return (
    <figure className={`figure-plate ${className ?? ""}`.trim()}>
      <button
        ref={triggerRef}
        className="figure-plate-frame"
        type="button"
        onClick={() => setOpen(true)}
        aria-label={`Enlarge ${label}: ${captionText}`}
      >
        <img
          src={src}
          alt={alt}
          width={width}
          height={height}
          loading="lazy"
          decoding="async"
        />
        <span className="figure-plate-zoom" aria-hidden="true">
          <ArrowsOut weight="light" />
          Enlarge
        </span>
      </button>

      <figcaption>
        <span>{label}</span>
        {caption}
      </figcaption>

      <AnimatePresence>
        {open && (
          <motion.div
            className="figure-lightbox-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduceMotion ? 0.08 : 0.26 }}
            onMouseDown={close}
          >
            <motion.div
              className="figure-lightbox"
              role="dialog"
              aria-modal="true"
              aria-label={`${label}, enlarged`}
              initial={reduceMotion ? false : { opacity: 0, y: 18, scale: 0.985 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.99 }}
              transition={{
                duration: reduceMotion ? 0 : 0.4,
                ease: [0.32, 0.72, 0, 1],
              }}
              onMouseDown={(event) => event.stopPropagation()}
            >
              <header>
                <strong className="figure-lightbox-label">{label}</strong>
                <div className="figure-lightbox-tools">
                  <button
                    type="button"
                    className="figure-lightbox-zoom"
                    onClick={() => setActualSize((value) => !value)}
                    aria-pressed={actualSize}
                  >
                    {actualSize ? (
                      <MagnifyingGlassMinus weight="light" />
                    ) : (
                      <MagnifyingGlassPlus weight="light" />
                    )}
                    {actualSize ? "Fit to screen" : "Actual size"}
                  </button>
                  <button
                    ref={closeRef}
                    className="figure-lightbox-close"
                    type="button"
                    onClick={close}
                    aria-label="Close the enlarged figure"
                  >
                    <X weight="light" />
                  </button>
                </div>
              </header>
              <div
                className="figure-lightbox-stage"
                data-size={actualSize ? "actual" : "fit"}
              >
                <img src={src} alt={alt} width={width} height={height} />
              </div>
              <figcaption className="figure-lightbox-caption">
                {captionText}
              </figcaption>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </figure>
  );
}
