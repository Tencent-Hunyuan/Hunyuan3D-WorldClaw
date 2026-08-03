import { Reveal } from "./Reveal";
import { SectionHeader } from "./SectionHeader";

/** Reproduces the paper abstract verbatim. */
export function Overview() {
  return (
    <section id="abstract" className="overview section-pad">
      <div className="page-shell">
        <SectionHeader index="02" eyebrow="Abstract" />

        <div className="abstract-layout">
          <Reveal className="abstract-lead" delay={0.06}>
            <p>
              Generating large-scale, freely explorable 3D worlds from
              open-ended text remains challenging because a system must jointly
              maintain <em>global spatial coherence</em>,{" "}
              <em>rich local content</em>, and <em>explicit assets</em> suitable
              for downstream editing and reuse.
            </p>
          </Reveal>
          <Reveal className="abstract-body" delay={0.12}>
            <p>
              We present WorldClaw, a fully agentic, coarse-to-fine framework
              for open-world 3D scene generation. Planning agents translate a
              text prompt into a structured specification of regions, terrain,
              assets, materials, and spatial relations. WorldClaw then builds a
              globally coherent terrain foundation from semantic layouts,
              reusable assets, generative or procedural materials, and a
              region-aware height field.
            </p>
            <p>
              For detail-demanding regions, it generates terrain-conditioned
              compositions, reconstructs editable textured meshes, and recovers
              their placement on the terrain; render-based agents further refine
              terrain, objects, appearance, and contacts. Across diverse
              open-world prompts, WorldClaw produces large-scale scenes with
              coherent spatial organization, visually compelling local content,
              and editable instance-level assets while preserving a consistent
              global terrain structure.
            </p>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
