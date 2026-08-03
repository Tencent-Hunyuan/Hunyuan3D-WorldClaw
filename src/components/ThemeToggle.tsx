import { MoonStars, SunDim } from "@phosphor-icons/react";
import { useState } from "react";

type Theme = "light" | "dark";

const themeColor: Record<Theme, string> = {
  light: "#f1efe8",
  dark: "#161613",
};

function currentTheme(): Theme {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(currentTheme);
  const nextTheme = theme === "light" ? "dark" : "light";

  const toggleTheme = () => {
    document.documentElement.dataset.theme = nextTheme;
    try {
      localStorage.setItem("worldclaw-theme", nextTheme);
    } catch {
      /* storage can be unavailable in private modes */
    }
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", themeColor[nextTheme]);
    setTheme(nextTheme);
  };

  return (
    <button
      className="theme-toggle"
      type="button"
      onClick={toggleTheme}
      aria-label={`Switch to the ${nextTheme} theme`}
      title={`Switch to the ${nextTheme} theme`}
    >
      {theme === "light" ? (
        <MoonStars weight="regular" aria-hidden="true" />
      ) : (
        <SunDim weight="regular" aria-hidden="true" />
      )}
    </button>
  );
}
