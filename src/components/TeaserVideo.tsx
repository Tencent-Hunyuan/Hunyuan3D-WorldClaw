import { useCallback, useEffect, useRef, useState } from "react";

export function TeaserVideo() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [paused, setPaused] = useState(false);
  const userPaused = useRef(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (userPaused.current) return;
        if (entry.isIntersecting) {
          video.play();
        } else {
          video.pause();
        }
      },
      { threshold: 0.3 },
    );

    observer.observe(video);
    return () => observer.disconnect();
  }, []);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused) {
      userPaused.current = false;
      video.play();
      setPaused(false);
    } else {
      userPaused.current = true;
      video.pause();
      setPaused(true);
    }
  }, []);

  return (
    <section className="teaser-video section-pad">
      <div className="page-shell">
        <div
          className={`teaser-video-wrap${paused ? " is-paused" : ""}`}
          onClick={togglePlay}
        >
          <video
            ref={videoRef}
            className="teaser-video-player"
            src={`${import.meta.env.BASE_URL}media/worldclaw_teaser.mp4`}
            muted
            loop
            playsInline
            preload="metadata"
          />
          <button
            className="teaser-video-toggle"
            aria-label={paused ? "Play video" : "Pause video"}
            type="button"
          >
            {paused ? (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
              </svg>
            ) : (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M6 4h4v16H6zm8 0h4v16h-4z" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </section>
  );
}
