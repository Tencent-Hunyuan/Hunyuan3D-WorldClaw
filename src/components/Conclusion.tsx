import { Code, GameController } from "@phosphor-icons/react";
import { Reveal } from "./Reveal";
import { SectionHeader } from "./SectionHeader";

const futureDirections = [
  {
    icon: Code,
    title: "Code-native 3D modeling",
    description:
      "Generative 3D backbones rarely recover explicit part hierarchies, parametric structure, articulation, or interaction logic. WorldClaw already authors terrain materials as executable Blender node graphs and shader scripts; extending that to object generation would make composition, material logic, adjustable parameters, and motion constraints explicitly editable.",
  },
  {
    icon: GameController,
    title: "Production engine integration",
    description:
      "Blender gives scriptable access to geometry, materials, and rendering, but large game worlds also need runtime procedural generation, navigation, physics, and interaction. Pairing WorldClaw's planning and code generation with engine-side procedural tooling would raise both scale and practical applicability.",
  },
];

export function Conclusion() {
  return (
    <section id="conclusion" className="conclusion section-pad">
      <div className="page-shell">
        <SectionHeader
          index="04"
          eyebrow="Conclusion"
          title="Structure first, detail second."
          lede={
            <p>
              Decoupling global world organization from local instance-level
              content is what lets a generated scene stay coherent as it grows.
            </p>
          }
        />

        <div className="conclusion-layout">
          <Reveal className="conclusion-statement" delay={0.08}>
            <p>
              WorldClaw is a coarse-to-fine agentic framework for building
              explicit, explorable, and editable 3D worlds from open-ended text.
              It turns intent into a structured specification, constructs a
              global terrain with region-aware semantics and multi-scale
              landforms, then generates objects only in the regions that call
              for fine-grained content.
            </p>
            <p>
              Render-guided agents refine terrain appearance, object quality,
              spatial arrangement, and object–terrain contact. Every world
              resolves to explicit terrain plus independently manageable
              textured meshes, which supports free-viewpoint exploration,
              object-level editing and reuse, and direct hand-off to rendering,
              animation-authoring, and game-engine workflows — balancing scene
              scale, visual quality, and editability across open-ended prompts.
            </p>
          </Reveal>

          <Reveal className="future-work" delay={0.16}>
            <h3>Future work</h3>
            <div>
              {futureDirections.map(({ icon: Icon, title, description }) => (
                <article key={title}>
                  <Icon weight="light" aria-hidden="true" />
                  <div>
                    <h4>{title}</h4>
                    <p>{description}</p>
                  </div>
                </article>
              ))}
            </div>
          </Reveal>

          <Reveal className="conclusion-coda" delay={0.22}>
            <span>Outlook</span>
            <p>
              As agents take over asset search, material graphs, and shader
              work, the central question of 3D content creation stops being how
              to construct every underlying component, and becomes{" "}
              <em>what kind of world the creator wishes to express.</em>
            </p>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
