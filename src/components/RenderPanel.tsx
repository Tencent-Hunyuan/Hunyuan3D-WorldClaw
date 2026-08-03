import { ArrowLeft, ArrowRight, ArrowsOut, X } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { type RefObject, useEffect, useRef, useState } from "react";
import {
  type Channel,
  channels,
  channelStill,
  channelVideo,
  type Scene,
} from "../data/content";
import { CropFrame } from "./CropFrame";

interface RenderPanelProps {
  scene: Scene;
  channel: Channel;
  playing: boolean;
  eager?: boolean;
  onExpand: () => void;
}

interface RenderLightboxProps {
  scene: Scene;
  channelIndex: number;
  onChannelChange: (index: number) => void;
  onClose: () => void;
}

function figureUrl(scene: Scene) {
  return `${import.meta.env.BASE_URL}${scene.image}`;
}

function videoUrl(scene: Scene, channel: Channel) {
  const path = channelVideo(scene, channel.id);
  return path ? `${import.meta.env.BASE_URL}${path}` : null;
}

/** Keeps a video element in sync with the section's play state. */
function useSyncedPlayback(
  ref: RefObject<HTMLVideoElement | null>,
  playing: boolean,
  ready: boolean,
) {
  useEffect(() => {
    const video = ref.current;
    if (!video || !ready) return;
    if (playing) {
      void video.play().catch(() => {
        /* autoplay can be refused; the still frame stays visible */
      });
    } else {
      video.pause();
    }
  }, [ref, playing, ready]);
}

/** Callers must key this on scene and channel so playback state resets. */
function ChannelMedia({
  scene,
  channel,
  playing,
  eager,
  controls = false,
}: {
  scene: Scene;
  channel: Channel;
  playing: boolean;
  eager?: boolean;
  controls?: boolean;
}) {
  const [ready, setReady] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const src = videoUrl(scene, channel);

  useSyncedPlayback(videoRef, playing, ready);

  return (
    <>
      <CropFrame
        src={figureUrl(scene)}
        alt={`${scene.name}, ${channel.label} channel`}
        rect={channelStill(channel.figureRow)}
        eager={eager}
      />
      {src && (
        <video
          key={src}
          ref={videoRef}
          className={`channel-video ${ready ? "is-ready" : ""}`}
          src={src}
          muted
          loop
          playsInline
          controls={controls}
          preload="metadata"
          onLoadedData={() => setReady(true)}
          onError={() => setReady(false)}
          aria-label={`${scene.name}, ${channel.label} channel clip`}
        />
      )}
    </>
  );
}

export function RenderPanel({
  scene,
  channel,
  playing,
  eager,
  onExpand,
}: RenderPanelProps) {
  return (
    <article className={`render-panel channel-${channel.id}`}>
      <button
        className="render-media"
        type="button"
        onClick={onExpand}
        aria-label={`Enlarge the ${channel.label} channel of ${scene.name}`}
      >
        <ChannelMedia
          scene={scene}
          channel={channel}
          playing={playing}
          eager={eager}
        />
        <span className="render-media-scrim" aria-hidden="true">
          <ArrowsOut weight="light" />
        </span>
      </button>
      <header className="render-panel-header">
        <strong>{channel.label}</strong>
        <span>{channel.detail}</span>
      </header>
    </article>
  );
}

export function RenderLightbox({
  scene,
  channelIndex,
  onChannelChange,
  onClose,
}: RenderLightboxProps) {
  const reduceMotion = useReducedMotion();
  const closeRef = useRef<HTMLButtonElement>(null);
  const channel = channels[channelIndex];

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key === "ArrowRight") {
        onChannelChange((channelIndex + 1) % channels.length);
      }
      if (event.key === "ArrowLeft") {
        onChannelChange((channelIndex + channels.length - 1) % channels.length);
      }
    };

    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", handleKeyDown);
    closeRef.current?.focus();

    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [channelIndex, onChannelChange, onClose]);

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
        aria-label={`${scene.name}, ${channel.label} channel, enlarged`}
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
            <span>{channel.label} channel</span>
          </div>
          <button
            className="render-lightbox-close"
            type="button"
            onClick={onClose}
            aria-label="Close enlarged rendering"
            ref={closeRef}
          >
            <X weight="light" />
          </button>
        </header>

        <div className="render-lightbox-stage">
          <ChannelMedia
            key={`${scene.id}-${channel.id}`}
            scene={scene}
            channel={channel}
            playing
            eager
            controls
          />
        </div>

        <footer>
          <button
            type="button"
            className="render-lightbox-step"
            onClick={() =>
              onChannelChange(
                (channelIndex + channels.length - 1) % channels.length,
              )
            }
            aria-label="Previous channel"
          >
            <ArrowLeft weight="light" />
          </button>

          <div className="render-lightbox-channels" role="tablist">
            {channels.map((item, index) => (
              <button
                key={item.id}
                type="button"
                role="tab"
                aria-selected={index === channelIndex}
                className={index === channelIndex ? "is-active" : ""}
                onClick={() => onChannelChange(index)}
              >
                {item.label}
              </button>
            ))}
          </div>

          <button
            type="button"
            className="render-lightbox-step"
            onClick={() => onChannelChange((channelIndex + 1) % channels.length)}
            aria-label="Next channel"
          >
            <ArrowRight weight="light" />
          </button>
        </footer>
      </motion.div>
    </motion.div>
  );
}
