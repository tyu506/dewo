export type StatePatch = {
  /** 与 SSE meta / 后端临时目录对应，用于拉取 infer 生成文件 */
  run_id?: string | null;
  dag_plan?: {
    graph_type?: string;
    nodes?: Array<Record<string, unknown>>;
    edges?: Array<{ source?: string; target?: string; edge_type?: string }>;
  };
  binding_by_node?: Record<string, { model_id?: string; prior_score?: unknown }>;
  node_outputs?: Record<string, unknown>;
  execution_by_node?: Record<string, Record<string, unknown>>;
  usage?: {
    totals?: { wall_sec?: number; llm?: { total_tokens?: number; prompt_tokens?: number; completion_tokens?: number } };
  };
  graph_eval?: {
    is_satisfied?: boolean;
    graph_error_type?: string;
    reason?: string | null;
    final_result?: unknown;
  };
  graph_final_message?: string | null;
  final_output_candidate?: unknown;
};

export type ExampleCard = {
  id: string;
  title: string;
  description: string;
  query: string;
  /** 中文卡片文案：通常来自 API；并与 `i18n/exampleZhOverlay.ts` 合并兜底（须与 backend examples_loader._EXAMPLE_ZH 一致） */
  title_zh?: string;
  description_zh?: string;
  query_zh?: string;
  /** 英文卡片标题后缀（「Example N: …」）；与 backend _EXAMPLE_EN_CARD_TITLE 对齐 */
  title_en?: string;
  inputs: Record<string, unknown>;
  expected_output_type?: string;
  split?: string;
  source_path?: string;
};
