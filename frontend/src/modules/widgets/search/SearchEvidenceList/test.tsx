import { GetSearchMessagesResponseDto } from "@api/dto/search";
import { searchMessagesResponse } from "@api/mock/responses/search";
import { mockServer } from "@api/mock/server";
import { ThemeProvider } from "@emotion/react";
import { theme } from "@provider/themeProvider";
import { getRouterPath, PATH_ROUTE } from "@routes/PATH_ROUTE";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { createMemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it } from "vitest";

import SearchEvidenceList from ".";

const WORKSPACE_ID = "1";
const CHAT_PATH = getRouterPath({
  routeKey: "CHAT",
  params: { workspaceId: WORKSPACE_ID },
});
const CONVERSATION_PATH = getRouterPath({
  routeKey: "CHAT_SESSION",
  params: { workspaceId: WORKSPACE_ID, sessionId: "10" },
});

const answers = new GetSearchMessagesResponseDto(
  searchMessagesResponse,
).items.filter(({ role }) => role === "ASSISTANT");
const [, answerWithThreeEvidences, answerWithoutEvidence] = answers;

const renderEvidenceList = async (initialPath: string) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const router = createMemoryRouter(
    [
      { path: PATH_ROUTE.CHAT, element: <SearchEvidenceList /> },
      { path: PATH_ROUTE.CHAT_SESSION, element: <SearchEvidenceList /> },
    ],
    { initialEntries: [initialPath] },
  );

  render(
    <ThemeProvider theme={theme}>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </ThemeProvider>,
  );

  // 메시지 조회가 끝나야 패널을 그릴지 정해져요
  await waitFor(() => expect(queryClient.isFetching()).toBe(0));
};

describe("SearchEvidenceList", () => {
  it("펼친 답변의 근거 문서를 서버가 준 순서대로 '찾은 기록'에 보여준다 (SEARCH-R2)", async () => {
    await renderEvidenceList(
      `${CONVERSATION_PATH}?messageId=${answerWithThreeEvidences.id}`,
    );

    const panel = screen.getByRole("region", { name: "찾은 기록" });
    const items = within(panel).getAllByRole("listitem");
    const expected = answerWithThreeEvidences.evidences;

    expect(items).toHaveLength(expected.length);
    items.forEach((item, index) => {
      expect(within(item).getByText(expected[index].title)).toBeInTheDocument();
    });
  });

  it("출처는 모두 '문서'로 보여주고, 서버가 아직 주지 않는 날짜는 보여주지 않는다 (근거 없음)", async () => {
    await renderEvidenceList(
      `${CONVERSATION_PATH}?messageId=${answerWithThreeEvidences.id}`,
    );

    const panel = screen.getByRole("region", { name: "찾은 기록" });
    const items = within(panel).getAllByRole("listitem");

    items.forEach((item) => {
      expect(within(item).getByText("문서")).toBeInTheDocument();
      expect(within(item).queryByText(/\d{4}/)).not.toBeInTheDocument();
    });
  });

  it("찾은 기록 카드는 그 문서 보기 화면으로 가는 링크다 (Figma Search/FoundRecords: 카드를 누르면 그 문서를 연다)", async () => {
    await renderEvidenceList(
      `${CONVERSATION_PATH}?messageId=${answerWithThreeEvidences.id}`,
    );

    const panel = screen.getByRole("region", { name: "찾은 기록" });
    const items = within(panel).getAllByRole("listitem");

    items.forEach((item, index) => {
      const { documentId } = answerWithThreeEvidences.evidences[index];

      expect(within(item).getByRole("link")).toHaveAttribute(
        "href",
        getRouterPath({
          routeKey: "DOCUMENT",
          params: { workspaceId: WORKSPACE_ID, documentId: String(documentId) },
        }),
      );
    });
  });

  it("네 번째 인용 문서도 제목과 링크가 있는 카드로 보여준다", async () => {
    const fourth = {
      documentId: 408,
      title: "레거시 독스 파일 구조 변경 및 백업",
      topic: "ADR",
      rank: 4,
    };
    mockServer.use(
      http.get(
        "*/api/v1/workspaces/:workspaceId/search/conversations/:conversationId/messages",
        () =>
          HttpResponse.json({
            ...searchMessagesResponse,
            items: searchMessagesResponse.items.map((message) =>
              message.id === answerWithThreeEvidences.id
                ? { ...message, evidences: [...message.evidences, fourth] }
                : message,
            ),
          }),
      ),
    );

    await renderEvidenceList(
      `${CONVERSATION_PATH}?messageId=${answerWithThreeEvidences.id}`,
    );

    const panel = screen.getByRole("region", { name: "찾은 기록" });
    const items = within(panel).getAllByRole("listitem");
    expect(items).toHaveLength(4);
    expect(
      within(items[3]).getByRole("link", { name: /레거시 독스 파일 구조 변경 및 백업/ }),
    ).toHaveAttribute(
      "href",
      getRouterPath({
        routeKey: "DOCUMENT",
        params: { workspaceId: WORKSPACE_ID, documentId: "408" },
      }),
    );
  });

  it("펼친 답변이 없으면 패널을 그리지 않는다 (SEARCH-R2)", async () => {
    await renderEvidenceList(CONVERSATION_PATH);

    expect(
      screen.queryByRole("region", { name: "찾은 기록" }),
    ).not.toBeInTheDocument();
  });

  it("펼친 답변에 근거가 없으면 패널을 그리지 않는다 (SEARCH-R2)", async () => {
    await renderEvidenceList(
      `${CONVERSATION_PATH}?messageId=${answerWithoutEvidence.id}`,
    );

    expect(
      screen.queryByRole("region", { name: "찾은 기록" }),
    ).not.toBeInTheDocument();
  });

  it("대화가 없는 새 대화에서는 패널을 그리지 않는다 (SEARCH-R2)", async () => {
    await renderEvidenceList(`${CHAT_PATH}?messageId=${answerWithThreeEvidences.id}`);

    expect(
      screen.queryByRole("region", { name: "찾은 기록" }),
    ).not.toBeInTheDocument();
  });
});
