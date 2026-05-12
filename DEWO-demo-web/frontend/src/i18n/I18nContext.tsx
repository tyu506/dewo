import { createContext, useContext } from "react";
import type { UILang } from "./messages";

export const I18nContext = createContext<{ lang: UILang }>({ lang: "zh" });

export function useI18n() {
  return useContext(I18nContext);
}
