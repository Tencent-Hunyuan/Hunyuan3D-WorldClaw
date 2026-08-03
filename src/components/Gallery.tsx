import {
  Buildings,
  CaretLeft,
  CaretRight,
  Fire,
  Flower,
  MapTrifold,
  Mountains,
  Pause,
  Play,
  Snowflake,
  Sun,
  Tree,
  Waves,
  Wind,
} from "@phosphor-icons/react";
import { AnimatePresence, useInView } from "motion/react";
import {
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  LAYOUT_HEIGHT,
  LAYOUT_WIDTH,
  channels,
  layoutRender,
  scenes,
} from "../data/content";
import { Reveal } from "./Reveal";
import { RenderLightbox, RenderPanel } from "./RenderPanel";
import { SectionHeader } from "./SectionHeader";

const sceneIcons = {
  flower: Flower,
  sun: Sun,
  leaf: Tree,
  snow: Snowflake,
  fire: Fire,
  waves: Waves,
  canyon: MapTrifold,
  tree: Tree,
  wind: Wind,
  ruins: Buildings,
};

type ViewerStyle = CSSProperties & {
  "--scene-accent": string;
};

export function Gallery() {
  const [sceneIndex, setSceneIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const [expandedChannel, setExpandedChannel] = useState<number | null>(null);
  const [railBounds, setRailBounds] = useState({ start: true, end: false });
  const railRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef({ pointerId: -1, startX: 0, scrollLeft: 0 });
  const didDragRef = useRef(false);
  const gridInView = useInView(gridRef, { amount: 0.15 });

  const activeScene = scenes[sceneIndex];
  const ActiveIcon = sceneIcons[activeScene.icon] ?? Mountains;
  const viewerStyle: ViewerStyle = { "--scene-accent": activeScene.accent };
  // Clips are optional per scene; without them the tiles stay on their stills
  // and the playback control would be a dead button.
  const hasClips = Boolean(activeScene.hasVideos);
  const playing = gridInView && !paused && expandedChannel === null;

  const updateRailBounds = useCallback(() => {
    const rail = railRef.current;
    if (!rail) return;

    const maxScroll = rail.scrollWidth - rail.clientWidth;
    setRailBounds({
      start: rail.scrollLeft <= 2,
      end: maxScroll <= 2 || rail.scrollLeft >= maxScroll - 2,
    });
  }, []);

  useEffect(() => {
    const rail = railRef.current;
    if (!rail) return;

    const frame = requestAnimationFrame(updateRailBounds);
    const observer = new ResizeObserver(updateRailBounds);
    observer.observe(rail);
    rail.addEventListener("scroll", updateRailBounds, { passive: true });

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      rail.removeEventListener("scroll", updateRailBounds);
    };
  }, [updateRailBounds]);

  const revealSceneChip = (index: number) => {
    const chip =
      railRef.current?.querySelectorAll<HTMLButtonElement>(".scene-chip")[index];
    chip?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
      inline: "center",
    });
  };

  const selectScene = (index: number) => {
    if (didDragRef.current) {
      didDragRef.current = false;
      return;
    }

    setSceneIndex(index);
    revealSceneChip(index);
  };

  const scrollRail = (direction: -1 | 1) => {
    const rail = railRef.current;
    if (!rail) return;
    rail.scrollBy({
      left: direction * rail.clientWidth * 0.72,
      behavior: "smooth",
    });
  };

  const handleRailPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.pointerType !== "mouse" || event.button !== 0) return;

    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      scrollLeft: event.currentTarget.scrollLeft,
    };
    didDragRef.current = false;
  };

  const handleRailPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (dragRef.current.pointerId !== event.pointerId) return;

    const delta = event.clientX - dragRef.current.startX;
    if (Math.abs(delta) > 4 && !didDragRef.current) {
      didDragRef.current = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      event.currentTarget.classList.add("is-dragging");
    }
    if (!didDragRef.current) return;

    event.preventDefault();
    event.currentTarget.scrollLeft = dragRef.current.scrollLeft - delta;
  };

  const endRailDrag = (event: PointerEvent<HTMLDivElement>) => {
    if (dragRef.current.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    event.currentTarget.classList.remove("is-dragging");
    dragRef.current.pointerId = -1;
    if (didDragRef.current) {
      window.setTimeout(() => {
        didDragRef.current = false;
      }, 0);
    }
  };

  const cancelRailDrag = (event: PointerEvent<HTMLDivElement>) => {
    endRailDrag(event);
    didDragRef.current = false;
  };

  const handleRailKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    const offset = event.key === "ArrowRight" ? 1 : scenes.length - 1;
    const next = (sceneIndex + offset) % scenes.length;
    setSceneIndex(next);
    const chip =
      event.currentTarget.querySelectorAll<HTMLButtonElement>(".scene-chip")[
        next
      ];
    chip?.focus();
    revealSceneChip(next);
  };

  return (
    <section id="results" className="results section-pad" style={viewerStyle}>
      <div className="page-shell">
        <SectionHeader
          index="01"
          eyebrow="Results"
          title="Eleven worlds, rendered four ways."
          lede={
            <p>
              Every world ships as an isometric layout plus four aligned render
              passes: appearance, instance masks, surface normals, and depth.
            </p>
          }
        />

        <Reveal className="scene-rail-shell" delay={0.06} amount={0.08}>
          <button
            className="scene-rail-nav is-prev"
            type="button"
            aria-label="Show previous worlds"
            disabled={railBounds.start}
            onClick={() => scrollRail(-1)}
          >
            <CaretLeft weight="bold" aria-hidden="true" />
          </button>
          <div className="scene-rail-wrap">
            <div
              ref={railRef}
              className="scene-rail"
              role="tablist"
              aria-label="Generated worlds"
              onKeyDown={handleRailKeys}
              onPointerDown={handleRailPointerDown}
              onPointerMove={handleRailPointerMove}
              onPointerUp={endRailDrag}
              onPointerCancel={cancelRailDrag}
            >
              {scenes.map((scene, index) => {
                const SceneIcon = sceneIcons[scene.icon] ?? Mountains;
                const isActive = index === sceneIndex;

                return (
                  <button
                    key={scene.id}
                    type="button"
                    role="tab"
                    aria-selected={isActive}
                    aria-controls="scene-detail"
                    tabIndex={isActive ? 0 : -1}
                    className={`scene-chip ${isActive ? "is-active" : ""}`}
                    onClick={() => selectScene(index)}
                    style={{ "--chip-accent": scene.accent } as CSSProperties}
                  >
                    <SceneIcon weight="light" aria-hidden="true" />
                    <span className="scene-chip-name">{scene.name}</span>
                    <span className="scene-chip-index">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
          <button
            className="scene-rail-nav is-next"
            type="button"
            aria-label="Show more worlds"
            disabled={railBounds.end}
            onClick={() => scrollRail(1)}
          >
            <CaretRight weight="bold" aria-hidden="true" />
          </button>
        </Reveal>

        <div
          id="scene-detail"
          role="tabpanel"
          aria-label={`${activeScene.name} renders`}
        >
          <Reveal className="scene-viewer" delay={0.08} amount={0.06}>
            <figure className="scene-layout">
              <div className="scene-layout-frame">
                <img
                  key={activeScene.id}
                  className="scene-layout-render"
                  src={`${import.meta.env.BASE_URL}${layoutRender(activeScene)}`}
                  alt={`Isometric layout of ${activeScene.name}`}
                  width={LAYOUT_WIDTH}
                  height={LAYOUT_HEIGHT}
                  loading={sceneIndex === 0 ? "eager" : "lazy"}
                  decoding="async"
                />
                <span className="scene-layout-tag">
                  <ActiveIcon weight="light" aria-hidden="true" />
                  {activeScene.type}
                </span>
              </div>
              <figcaption>
                <span>Layout</span>
                Full-world isometric overview of the generated terrain and its
                placed instances.
              </figcaption>
            </figure>

            <div className="scene-brief">
              <span className="scene-brief-label">Prompt</span>
              <blockquote>{activeScene.prompt}</blockquote>
              <dl className="scene-brief-meta">
                <div>
                  <dt>World</dt>
                  <dd>
                    {String(sceneIndex + 1).padStart(2, "0")} / {scenes.length}
                  </dd>
                </div>
                <div>
                  <dt>Family</dt>
                  <dd>{activeScene.type}</dd>
                </div>
                <div>
                  <dt>Channels</dt>
                  <dd>{channels.length} aligned</dd>
                </div>
              </dl>
            </div>
          </Reveal>

          <Reveal className="channel-block" delay={0.06} amount={0.05}>
            <div className="channel-toolbar">
              <p>
                Channel renders
                <span>{activeScene.name}</span>
              </p>
              {hasClips && (
                <button
                  className="playback-toggle"
                  type="button"
                  onClick={() => setPaused((value) => !value)}
                  aria-pressed={paused}
                >
                  {paused ? (
                    <Play weight="fill" aria-hidden="true" />
                  ) : (
                    <Pause weight="fill" aria-hidden="true" />
                  )}
                  {paused ? "Play clips" : "Pause clips"}
                </button>
              )}
            </div>

            <div className="render-grid" ref={gridRef}>
              {channels.map((channel, index) => (
                <RenderPanel
                  key={`${activeScene.id}-${channel.id}`}
                  scene={activeScene}
                  channel={channel}
                  playing={playing}
                  eager={sceneIndex === 0 && index === 0}
                  onExpand={() => setExpandedChannel(index)}
                />
              ))}
            </div>
          </Reveal>
        </div>
      </div>

      <AnimatePresence>
        {expandedChannel !== null && (
          <RenderLightbox
            scene={activeScene}
            channelIndex={expandedChannel}
            onChannelChange={setExpandedChannel}
            onClose={() => setExpandedChannel(null)}
          />
        )}
      </AnimatePresence>
    </section>
  );
}
