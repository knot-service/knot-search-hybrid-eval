/**
 * 탐색 DTO
 *
 * - GET  /api/v1/workspaces/{workspaceId}/search/conversations
 * - POST /api/v1/workspaces/{workspaceId}/search/conversations (SSE)
 * - POST /api/v1/workspaces/{workspaceId}/search/conversations/{conversationId}/questions (SSE)
 * - POST /api/v1/workspaces/{workspaceId}/search/conversations/{conversationId}/questions/{questionId}/retry (SSE)
 * - GET  /api/v1/workspaces/{workspaceId}/search/conversations/{conversationId}/messages
 */

/** 답변 준비 단계. SEARCHING은 문서 검색 중, GENERATING은 답변 생성 중 */
export type SearchStreamStage = "SEARCHING" | "GENERATING";

/** 메시지를 쓴 쪽. USER는 질문, ASSISTANT는 답변 */
export type SearchMessageRole = "USER" | "ASSISTANT";

/** 메시지 상태. 질문은 RECEIVED, 답변은 STREAMING에서 COMPLETED·FAILED·STOPPED 중 하나로 끝남 */
export type SearchMessageStatus =
  | "RECEIVED"
  | "STREAMING"
  | "COMPLETED"
  | "FAILED"
  | "STOPPED";

/**
 * 답변 근거 문서의 서버 응답 모양. 메시지 조회의 evidences[]와 evidence 이벤트의 items[]가 공유.
 *
 * 임시 가정: evidence 이벤트의 items[]는 API 명세에 필드가 아직 정해지지 않아 메시지 조회의 근거 필드를 따라 둠.
 * 명세가 정해지면 갱신
 */
export interface SearchEvidenceRaw {
  documentId: number;
  title: string;
  topic: string;
  rank: number;
}

/** 답변의 근거가 된 문서 하나. 메시지 조회와 evidence 이벤트가 공유 */
export class SearchEvidenceDto {
  /** 근거 문서 ID */
  documentId: number;
  /** 문서 제목 */
  title: string;
  /** 문서 주제 */
  topic: string;
  /** 관련도 순위. 1~3이고 작을수록 관련도가 높음 */
  rank: number;

  constructor(raw: SearchEvidenceRaw) {
    this.documentId = raw.documentId;
    this.title = raw.title;
    this.topic = raw.topic;
    this.rank = raw.rank;
  }
}

// GET /api/v1/workspaces/{workspaceId}/search/conversations

/** 대화 목록 한 줄의 서버 응답 모양 */
export interface SearchConversationRaw {
  id: number;
  title: string;
  lastMessagePreview: string | null;
  createdAt: string;
  updatedAt: string;
}

/** 대화 목록에 놓이는 탐색 대화 하나 */
export class SearchConversationDto {
  /** 탐색 대화 ID */
  id: number;
  /** 첫 질문에서 만든 표시 제목 */
  title: string;
  /** 마지막 메시지 미리보기. 메시지가 없으면 null */
  lastMessagePreview: string | null;
  /** 첫 질문을 접수한 시각(ISO 8601) */
  createdAt: string;
  /** 마지막으로 메시지가 오간 시각(ISO 8601). 목록은 이 값의 내림차순 */
  updatedAt: string;

  constructor(raw: SearchConversationRaw) {
    this.id = raw.id;
    this.title = raw.title;
    this.lastMessagePreview = raw.lastMessagePreview;
    this.createdAt = raw.createdAt;
    this.updatedAt = raw.updatedAt;
  }
}

/** 대화 목록 조회의 서버 응답 모양 */
export interface GetSearchConversationsResponseRaw {
  items: SearchConversationRaw[];
  nextCursor: string | null;
}

/** 내 탐색 대화 목록 한 페이지 조회 응답 */
export class GetSearchConversationsResponseDto {
  /** 대화 목록. updatedAt 내림차순, 같으면 ID 내림차순. 빈 대화와 첫 답변이 실패한 대화는 서버가 빼요 */
  items: SearchConversationDto[];
  /** 다음 페이지를 받을 커서. 마지막 페이지면 null */
  nextCursor: string | null;

  constructor(raw: GetSearchConversationsResponseRaw) {
    this.items = raw.items.map((item) => new SearchConversationDto(item));
    this.nextCursor = raw.nextCursor;
  }
}

// POST /api/v1/workspaces/{workspaceId}/search/conversations (SSE)

/** 첫 질문 전송 시 앱이 넘기는 값 */
export interface PostSearchConversationRequestInput {
  content: string;
  requestId: string;
}

/** 첫 질문 전송 요청 본문. 응답은 JSON이 아니라 SSE 스트림 */
export class PostSearchConversationRequestDto {
  /** 질문 내용. 앞뒤 공백 제거 */
  content: string;
  /** 같은 질문의 재전송을 서버가 알아보는 키(UUID). 재시도에도 같은 값을 보냄 */
  requestId: string;

  constructor({ content, requestId }: PostSearchConversationRequestInput) {
    this.content = content.trim();
    this.requestId = requestId;
  }
}

// POST /api/v1/workspaces/{workspaceId}/search/conversations/{conversationId}/questions (SSE)

/** 후속 질문 전송 시 앱이 넘기는 값 */
export interface PostSearchQuestionRequestInput {
  content: string;
  requestId: string;
}

/** 후속 질문 전송 요청 본문. 첫 질문과 모양이 같고 응답도 같은 SSE 스트림 */
export class PostSearchQuestionRequestDto {
  /** 질문 내용. 앞뒤 공백 제거 */
  content: string;
  /** 같은 질문의 재전송을 서버가 알아보는 키(UUID). 재시도에도 같은 값을 보냄 */
  requestId: string;

  constructor({ content, requestId }: PostSearchQuestionRequestInput) {
    this.content = content.trim();
    this.requestId = requestId;
  }
}

// POST /api/v1/workspaces/{workspaceId}/search/conversations/{conversationId}/questions/{questionId}/retry (SSE)

/** 실패한 답변 재시도 시 앱이 넘기는 값 */
export interface PostSearchAnswerRetryRequestInput {
  requestId: string;
}

/**
 * 실패한 답변 재시도 요청 본문. 질문 내용은 경로의 질문을 다시 쓰므로 싣지 않아요.
 * 응답은 첫 질문과 같은 SSE 스트림이고, accepted는 경로의 질문 ID와 실패했던 답변 메시지 ID를 다시 담아요
 */
export class PostSearchAnswerRetryRequestDto {
  /** 같은 재시도 요청의 중복 실행을 서버가 알아보는 키(UUID) */
  requestId: string;

  constructor({ requestId }: PostSearchAnswerRetryRequestInput) {
    this.requestId = requestId;
  }
}

// 아래 이벤트 DTO는 첫 질문·후속 질문·재시도 스트림이 공유

/** `event: accepted`의 data 모양 */
export interface SearchStreamAcceptedRaw {
  conversationId: number;
  questionMessageId: number;
  answerMessageId: number;
}

/** 서버가 질문을 받아 대화와 메시지를 만들었음을 알리는 이벤트. 스트림의 첫 이벤트 */
export class SearchStreamAcceptedDto {
  /** 탐색 대화 ID. 첫 질문이면 새로 만들어진 대화, 후속 질문이면 경로의 대화 */
  conversationId: number;
  /** 저장된 질문 메시지 ID */
  questionMessageId: number;
  /** 이어서 채워질 답변 메시지 ID. 뒤따르는 이벤트의 answerMessageId와 같음 */
  answerMessageId: number;

  constructor(raw: SearchStreamAcceptedRaw) {
    this.conversationId = raw.conversationId;
    this.questionMessageId = raw.questionMessageId;
    this.answerMessageId = raw.answerMessageId;
  }
}

/** `event: progress`의 data 모양 */
export interface SearchStreamProgressRaw {
  stage: SearchStreamStage;
}

/** 답변 준비 단계가 바뀌었음을 알리는 이벤트 */
export class SearchStreamProgressDto {
  /** 지금 단계. SEARCHING(문서 검색 중) 또는 GENERATING(답변 생성 중) */
  stage: SearchStreamStage;

  constructor(raw: SearchStreamProgressRaw) {
    this.stage = raw.stage;
  }
}

/** `event: delta`의 data 모양 */
export interface SearchStreamDeltaRaw {
  answerMessageId: number;
  text: string;
}

/** 답변 조각. 스트림 하나에서 여러 번 도착함 */
export class SearchStreamDeltaDto {
  /** 이 조각이 속한 답변 메시지 ID */
  answerMessageId: number;
  /** 답변의 일부 텍스트. 도착 순서대로 이어 붙이면 전체 답변이 됨 */
  text: string;

  constructor(raw: SearchStreamDeltaRaw) {
    this.answerMessageId = raw.answerMessageId;
    this.text = raw.text;
  }
}

/** `event: evidence`의 data 모양 */
export interface SearchStreamEvidenceRaw {
  answerMessageId: number;
  items: SearchEvidenceRaw[];
}

/** 답변의 근거 문서 목록을 알리는 이벤트 */
export class SearchStreamEvidenceDto {
  /** 근거가 붙는 답변 메시지 ID */
  answerMessageId: number;
  /** 답변에 저장된 근거 문서. rank 오름차순 */
  items: SearchEvidenceDto[];

  constructor(raw: SearchStreamEvidenceRaw) {
    this.answerMessageId = raw.answerMessageId;
    this.items = raw.items.map((item) => new SearchEvidenceDto(item));
  }
}

/** `event: excluded_documents`의 data 모양 */
export interface SearchStreamExcludedDocumentsRaw {
  count: number;
}

/** 답변 근거에서 제외된 문서 수를 알리는 이벤트 */
export class SearchStreamExcludedDocumentsDto {
  /** 제외된 문서 수 */
  count: number;

  constructor(raw: SearchStreamExcludedDocumentsRaw) {
    this.count = raw.count;
  }
}

/** `event: completed`의 data 모양 */
export interface SearchStreamCompletedRaw {
  answerMessageId: number;
  status: "COMPLETED";
}

/** 답변이 끝나고 저장까지 마쳤음을 알리는 이벤트. 스트림의 마지막 이벤트 */
export class SearchStreamCompletedDto {
  /** 완료된 답변 메시지 ID */
  answerMessageId: number;
  /** 답변 메시지의 최종 상태. 항상 COMPLETED */
  status: "COMPLETED";

  constructor(raw: SearchStreamCompletedRaw) {
    this.answerMessageId = raw.answerMessageId;
    this.status = raw.status;
  }
}

/** `event: failed`의 data 모양 */
export interface SearchStreamFailedRaw {
  answerMessageId: number;
  status: "FAILED";
  code: string;
}

/** accepted 뒤에 답변 생성이 실패했음을 알리는 이벤트. HTTP 상태가 아니라 이벤트로 옴 */
export class SearchStreamFailedDto {
  /** 실패한 답변 메시지 ID */
  answerMessageId: number;
  /** 답변 메시지의 최종 상태. 항상 FAILED */
  status: "FAILED";
  /** 실패 원인 오류 코드. 예: LLM_STREAM_FAILED */
  code: string;

  constructor(raw: SearchStreamFailedRaw) {
    this.answerMessageId = raw.answerMessageId;
    this.status = raw.status;
    this.code = raw.code;
  }
}

/** `event: stopped`의 data 모양 */
export interface SearchStreamStoppedRaw {
  answerMessageId: number;
  status: "STOPPED";
}

/** 답변 생성이 중단됐음을 알리는 이벤트 */
export class SearchStreamStoppedDto {
  /** 중단된 답변 메시지 ID */
  answerMessageId: number;
  /** 답변 메시지의 최종 상태. 항상 STOPPED */
  status: "STOPPED";

  constructor(raw: SearchStreamStoppedRaw) {
    this.answerMessageId = raw.answerMessageId;
    this.status = raw.status;
  }
}

/**
 * 탐색 스트림이 흘려보내는 이벤트 한 건.
 *
 * `event`로 갈라 `data`의 모양이 좁혀지도록 유니언으로 둡니다.
 * 명세에 없는 이벤트 이름은 요청 함수에서 버리므로 여기에 두지 않습니다.
 */
export type SearchStreamEvent =
  | { event: "accepted"; data: SearchStreamAcceptedDto }
  | { event: "progress"; data: SearchStreamProgressDto }
  | { event: "delta"; data: SearchStreamDeltaDto }
  | { event: "evidence"; data: SearchStreamEvidenceDto }
  | { event: "excluded_documents"; data: SearchStreamExcludedDocumentsDto }
  | { event: "completed"; data: SearchStreamCompletedDto }
  | { event: "failed"; data: SearchStreamFailedDto }
  | { event: "stopped"; data: SearchStreamStoppedDto };

// GET /api/v1/workspaces/{workspaceId}/search/conversations/{conversationId}/messages

/** 탐색 메시지 하나의 서버 응답 모양 */
export interface SearchMessageRaw {
  id: number;
  role: SearchMessageRole;
  sequence: number;
  content: string;
  status: SearchMessageStatus;
  createdAt: string;
  evidences: SearchEvidenceRaw[];
}

/** 탐색 대화의 메시지 하나 */
export class SearchMessageDto {
  /** 메시지 ID. accepted 이벤트의 questionMessageId·answerMessageId와 같은 값 */
  id: number;
  /** 쓴 쪽. USER는 질문, ASSISTANT는 답변 */
  role: SearchMessageRole;
  /** 대화 안에서의 순서. 응답은 오름차순 */
  sequence: number;
  /** 질문 내용 또는 답변 본문. 생성 중인 답변은 저장된 데까지의 본문 */
  content: string;
  /** 메시지 상태. 질문은 RECEIVED */
  status: SearchMessageStatus;
  /** 만든 시각(ISO 8601) */
  createdAt: string;
  /** 답변에 저장된 근거 문서. rank 오름차순. 질문(USER)은 빈 배열 */
  evidences: SearchEvidenceDto[];

  constructor(raw: SearchMessageRaw) {
    this.id = raw.id;
    this.role = raw.role;
    this.sequence = raw.sequence;
    this.content = raw.content;
    this.status = raw.status;
    this.createdAt = raw.createdAt;
    this.evidences = raw.evidences.map(
      (evidence) => new SearchEvidenceDto(evidence),
    );
  }
}

/** 탐색 메시지 조회의 서버 응답 모양 */
export interface GetSearchMessagesResponseRaw {
  conversationId: number;
  items: SearchMessageRaw[];
  hasPrevious: boolean;
  previousCursor: string | null;
}

/** 탐색 대화의 최근 메시지 한 페이지 조회 응답 */
export class GetSearchMessagesResponseDto {
  /** 조회한 탐색 대화 ID */
  conversationId: number;
  /** 메시지 목록. sequence 오름차순(오래된 것부터) */
  items: SearchMessageDto[];
  /** 이 페이지보다 이전 메시지가 더 있는지 */
  hasPrevious: boolean;
  /** 이전 페이지를 받을 커서. 이전 메시지가 없으면 null */
  previousCursor: string | null;

  constructor(raw: GetSearchMessagesResponseRaw) {
    this.conversationId = raw.conversationId;
    this.items = raw.items.map((item) => new SearchMessageDto(item));
    this.hasPrevious = raw.hasPrevious;
    this.previousCursor = raw.previousCursor;
  }
}
