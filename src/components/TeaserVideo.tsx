import { ArrowsOut, Pause, Play, X } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import {
  type ChangeEvent,
  type CSSProperties,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { Reveal } from "./Reveal";

const CLIP = "media/worldclaw-teaser.mp4";
const POSTER = "media/worldclaw-teaser.webp";

/* Stands in until the metadata arrives, which `preload="none"` defers until
   playback starts — so this is on screen for most readers and has to match the
   encoded clip. */
const CLIP_SECONDS = 91;

/*
 * The clip is the heaviest thing on the page that nobody asked for, and
 * `preload="none"` means play() is what fetches it. Readers who have told the
 * browser to hold back get the poster and a play button instead.
 */
function shouldAutoplay() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return false;
  }
  const connection = (
    navigator as Navigator & { connection?: { saveData?: boolean } }
  ).connection;
  return !connection?.saveData;
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00";
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export function TeaserVideo() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [expanded, setExpanded] = useState(false);
  /* A manual pause has to survive scrolling away and back, or the observer
     would restart the clip the moment it returns to view. */
  const pausedByUser = useRef(false);

  const url = (path: string) => `${import.meta.env.BASE_URL}${path}`;

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          if (pausedByUser.current || expanded || !shouldAutoplay()) return;
          void video.play().catch(() => {
            /* autoplay can be refused; the poster stays up */
          });
        } else {
          video.pause();
        }
      },
      { threshold: 0.35 },
    );

    observer.observe(video);
    return () => observer.disconnect();
  }, [expanded]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused) {
      pausedByUser.current = false;
      void video.play().catch(() => {});
    } else {
      pausedByUser.current = true;
      video.pause();
    }
  }, []);

  const seek = (event: ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video || !duration) return;
    video.currentTime = (Number(event.target.value) / 1000) * duration;
  };

  // Hands the enlarged copy the frame the inline one was on, and takes it back
  // on close, so the clip never restarts from the title card.
  const openExpanded = () => {
    videoRef.current?.pause();
    setExpanded(true);
  };

  const closeExpanded = (at: number) => {
    setExpanded(false);
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = at;
    if (!pausedByUser.current) void video.play().catch(() => {});
  };

  const progress = duration ? time / duration : 0;

  return (
    <section className="teaser" aria-label="Overview video">
      <div className="page-shell">
        <Reveal className="teaser-rail">
          <span className="section-eyebrow">Overview</span>
          <span className="section-rule" aria-hidden="true" />
          <span className="teaser-rail-meta">
            {formatTime(duration || CLIP_SECONDS)}
          </span>
        </Reveal>

        <Reveal className="teaser-layout" delay={0.06}>
          <div className={`teaser-frame ${playing ? "" : "is-paused"}`.trim()}>
            <video
              ref={videoRef}
              className="teaser-player"
              src={url(CLIP)}
              poster={url(POSTER)}
              muted
              loop
              playsInline
              preload="none"
              tabIndex={-1}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
              onLoadedMetadata={(event) =>
                setDuration(event.currentTarget.duration)
              }
            />

            <div className="teaser-controls">
              <button
                className="teaser-play"
                type="button"
                onClick={togglePlay}
                aria-label={playing ? "Pause the overview" : "Play the overview"}
              >
                {playing ? (
                  <Pause weight="fill" aria-hidden="true" />
                ) : (
                  <Play weight="fill" aria-hidden="true" />
                )}
              </button>

              <input
                className="teaser-scrub"
                style={{ "--played": `${progress * 100}%` } as CSSProperties}
                type="range"
                min={0}
                max={1000}
                step={1}
                value={Math.round(progress * 1000)}
                onChange={seek}
                aria-label="Seek through the overview"
                aria-valuetext={`${formatTime(time)} of ${formatTime(duration)}`}
              />

              <span className="teaser-time">
                {formatTime(time)}
                <i aria-hidden="true">/</i>
                {formatTime(duration)}
              </span>

              <button
                className="teaser-expand"
                type="button"
                onClick={openExpanded}
                aria-label="Enlarge the overview"
              >
                <ArrowsOut weight="light" aria-hidden="true" />
              </button>
            </div>
          </div>

          <div className="teaser-note">
            <h2>A world from a single sentence.</h2>
            <p>
              WorldClaw takes one block of open-ended text and returns a
              high-quality 3D scene you can explore, with the terrain and every
              object standing in it kept as separate, editable instances.
            </p>
            <p className="teaser-note-hint">
              Four worlds — a snowline village, a canyon settlement, a tropical
              island, and an arctic outpost — show the range one pipeline
              covers. The object library each draws on, and the instance pass
              behind it, show that every mesh stays separate and editable.
            </p>
          </div>
        </Reveal>
      </div>

      <AnimatePresence>
        {expanded && (
          <TeaserLightbox
            src={url(CLIP)}
            poster={url(POSTER)}
            startAt={time}
            onClose={closeExpanded}
          />
        )}
      </AnimatePresence>
    </section>
  );
}

interface TeaserLightboxProps {
  src: string;
  poster: string;
  startAt: number;
  onClose: (at: number) => void;
}

function TeaserLightbox({ src, poster, startAt, onClose }: TeaserLightboxProps) {
  const reduceMotion = useReducedMotion();
  const videoRef = useRef<HTMLVideoElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const timeRef = useRef(startAt);

  const dismiss = useCallback(() => {
    onClose(videoRef.current?.currentTime ?? timeRef.current);
  }, [onClose]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") dismiss();
    };

    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", onKeyDown);
    closeRef.current?.focus();

    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [dismiss]);

  return (
    <motion.div
      className="render-lightbox-backdrop"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: reduceMotion ? 0.08 : 0.26 }}
      onMouseDown={dismiss}
    >
      <motion.div
        className="render-lightbox"
        role="dialog"
        aria-modal="true"
        aria-label="WorldClaw overview, enlarged"
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
            <strong>WorldClaw</strong>
            <span>Overview</span>
          </div>
          <button
            ref={closeRef}
            className="render-lightbox-close"
            type="button"
            onClick={dismiss}
            aria-label="Close the enlarged overview"
          >
            <X weight="light" />
          </button>
        </header>

        <div className="render-lightbox-stage">
          <video
            ref={videoRef}
            className="teaser-lightbox-player"
            src={src}
            poster={poster}
            muted
            loop
            playsInline
            controls
            autoPlay
            onLoadedMetadata={(event) => {
              event.currentTarget.currentTime = startAt;
            }}
          />
        </div>
      </motion.div>
    </motion.div>
  );
}
