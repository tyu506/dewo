/**
 * 与 backend/app/examples_loader.py 中 _EXAMPLE_ZH 保持逐字一致（前端兜底合并）。
 * 修改后端时请同步更新此文件。
 */
import type { ExampleCard } from "../types";

export const EXAMPLE_ZH_OVERLAY: Record<
  string,
  Pick<ExampleCard, "title_zh" | "description_zh" | "query_zh" | "title_en">
> = {
  req_graph_000011: {
    title_en: "Conference recording — transcription, NER & notes alignment",
    title_zh: "会议整场录音转写、命名实体与笔记对齐",
    description_zh: "复杂 · 自动语音识别、词元分类、句子相似度",
    query_zh:
      "我最近参加了一场会议，并用文件名「graph_011_audio_1.wav」录下了整场会议。" +
      "我有些想不起来会议期间讨论的所有重要细节了。你能帮我转写这段音频吗？" +
      "另外，如果能在转写中标出（高亮）所有命名实体就太好了。" +
      "我在会议期间记了一些笔记（「My Conference Notes／我的会议笔记」），" +
      "你能帮我理解我的笔记与实际转写内容在多大程度上是一致的吗？" +
      "我的会议笔记如下：「会议主要讨论了 Apollo 项目的当前进展。" +
      "张伟提到，核心模块需要在三月底之前完成，并且应与市场部协调测试资源，以确保按期交付。」",
  },
  req_graph_000021: {
    title_en: "Image — scene description, object detection & structured English report",
    title_zh: "图像场景描述、目标检测与英文结构化报告",
    description_zh: "复杂 · 图像转文本、目标检测、摘要",
    query_zh:
      "我有一张名为「graph_021_image_1.png」的图片。" +
      "首先，请描述图像中的整体场景。" +
      "同时，请检测并列出图像中出现的主要物体及其标签。" +
      "最后，请将上述两部分结果合并为一份结构化的英文报告，其中包含两个小节：" +
      "「Scene Description」和「Detected Objects」。",
  },
};

export function mergeExampleZhFromOverlay(ex: ExampleCard): ExampleCard {
  const o = EXAMPLE_ZH_OVERLAY[ex.id];
  if (!o) return ex;
  return {
    ...ex,
    title_en: ex.title_en ?? o.title_en,
    title_zh: ex.title_zh ?? o.title_zh,
    description_zh: ex.description_zh ?? o.description_zh,
    query_zh: ex.query_zh ?? o.query_zh,
  };
}
