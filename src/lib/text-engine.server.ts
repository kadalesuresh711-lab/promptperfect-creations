/**
 * The ONLY text engine in this app: Agnes AI (agnes-2.5-flash).
 */

import { agnesChat } from "./agnes.server";

export async function textChat(
  system: string,
  user: string,
  opts: {
    temperature?: number;
    maxOutputTokens?: number;
    timeoutMs?: number;
    attempts?: number;
  } = {},
): Promise<string> {
  return agnesChat(user, { system, ...opts });
}
