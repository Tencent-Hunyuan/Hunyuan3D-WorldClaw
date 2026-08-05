import { ArrowLeft, ArrowRight, ArrowsOut, X } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import {
  type Scene,
  VIEW_FULL_HEIGHT,
  VIEW_FULL_WIDTH,
  VIEW_SHOTS_PER_TRACK,
  VIEW_TILE_HEIGHT,
  VIEW_TILE_WIDTH,
  sceneViewSequence,
  sceneViews,
  viewTracks,
} from "../data/content";

interface SceneViewsProps {
  scene: Scene;
  /** First tile of the first scene is above the fold, so it skips lazy loading. */
  eager?: boolean;
}

function assetUrl(path: string) {
  return `${import.meta.env.BASE_URL}${path}`;
}

/**
 * The two camera tracks for one world, three shots each.
 *
 * Tiles are 640px derivatives — six of them next to the layout never exceed a
 * quarter of the column — and the 1600px frame is only requested once someone
 * opens the lightbox.
 */
export function SceneViews({ scene, eager }: SceneViewsProps) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const sequence = sceneViewSequence(scene);

  return (
    <div className="scene-views">
      {viewTracks.map((track, trackIndex) => (
        <section
          key={track.id}
          className={`view-track view-track-${track.id}`}
          aria-label={`${track.label} views of ${scene.name}`}
        >
          <header className="view-track-head">
            <span className="view-track-label">{track.label}</span>
            <p>{track.detail}</p>
          </header>

          <div className="view-track-grid">
            {sceneViews(scene, track).map((view, index) => (
              <button
                key={`${scene.id}-${view.id}`}
                type="button"
                className="view-tile"
                onClick={() =>
                  setOpenIndex(trackIndex * VIEW_SHOTS_PER_TRACK + index)
                }
                aria-label={`Enlarge ${view.alt}`}
              >
                <img
                  src={assetUrl(view.tile)}
                  alt={view.alt}
                  width={VIEW_TILE_WIDTH}
                  height={VIEW_TILE_HEIGHT}
                  loading={eager && trackIndex === 0 && index === 0 ? "eager" : "lazy"}
                  decoding="async"
                />
                <span className="view-tile-scrim" aria-hidden="true">
                  <ArrowsOut weight="light" />
                </span>
                <span className="view-tile-index" aria-hidden="true">
                  {String(view.shot).padStart(2, "0")}
                </span>
              </button>
            ))}
          </div>
        </section>
      ))}

      <AnimatePresence>
        {openIndex !== null && (
          <ViewLightbox
            scene={scene}
            views={sequence}
            index={openIndex}
            onIndexChange={setOpenIndex}
            onClose={() => setOpenIndex(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

interface ViewLightboxProps {
  scene: Scene;
  views: ReturnType<typeof sceneViewSequence>;
  index: number;
  onIndexChange: (index: number) => void;
  onClose: () => void;
}

function ViewLightbox({
  scene,
  views,
  index,
  onIndexChange,
  onClose,
}: ViewLightboxProps) {
  const reduceMotion = useReducedMotion();
  const closeRef = useRef<HTMLButtonElement>(null);
  const view = views[index];

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key === "ArrowRight") {
        onIndexChange((index + 1) % views.length);
      }
      if (event.key === "ArrowLeft") {
        onIndexChange((index + views.length - 1) % views.length);
      }
    };

    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", handleKeyDown);
    closeRef.current?.focus();

    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [index, onClose, onIndexChange, views.length]);

  return (
    <motion.div
      className="render-lightbox-backdrop"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: reduceMotion ? 0.08 : 0.26 }}
      onMouseDown={onClose}
    >
      <motion.div
        className="render-lightbox"
        role="dialog"
        aria-modal="true"
        aria-label={`${view.alt}, enlarged`}
        initial={reduceMotion ? false : { opacity: 0, y: 20, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 10, scale: 0.99 }}
        transition={{
          duration: reduceMotion ? 0 : 0.4,
          ease: [0.32, 0.72, 0, 1],
        }}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div className="render-lightbox-title">
            <strong>{scene.name}</strong>
            <span>
              {view.track.label} view {view.shot}
            </span>
          </div>
          <button
            ref={closeRef}
            className="render-lightbox-close"
            type="button"
            onClick={onClose}
            aria-label="Close enlarged view"
          >
            <X weight="light" />
          </button>
        </header>

        <div className="render-lightbox-stage is-still">
          <img
            key={view.full}
            src={assetUrl(view.full)}
            alt={view.alt}
            width={VIEW_FULL_WIDTH}
            height={VIEW_FULL_HEIGHT}
          />
        </div>

        <footer>
          <button
            type="button"
            className="render-lightbox-step"
            onClick={() => onIndexChange((index + views.length - 1) % views.length)}
            aria-label="Previous view"
          >
            <ArrowLeft weight="light" />
          </button>

          <div className="view-lightbox-nav">
            {viewTracks.map((track) => (
              <div key={track.id} className="view-lightbox-group">
                <span>{track.label}</span>
                {views.map((item, itemIndex) =>
                  item.track.id === track.id ? (
                    <button
                      key={item.id}
                      type="button"
                      className={itemIndex === index ? "is-active" : ""}
                      aria-current={itemIndex === index}
                      aria-label={`${track.label} view ${item.shot}`}
                      onClick={() => onIndexChange(itemIndex)}
                    >
                      {item.shot}
                    </button>
                  ) : null,
                )}
              </div>
            ))}
          </div>

          <button
            type="button"
            className="render-lightbox-step"
            onClick={() => onIndexChange((index + 1) % views.length)}
            aria-label="Next view"
          >
            <ArrowRight weight="light" />
          </button>
        </footer>
      </motion.div>
    </motion.div>
  );
}
