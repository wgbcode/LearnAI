import {alovaInstance} from "@/utils/request.ts";

// 测试
export const test = async (params: { content: string }) => {
  const res = await alovaInstance.Post('/api/v1/chat', params);
  return res
}
