import type { SearchMessage } from "@/shared/types/searchAnswer";

/** 턴의 답변. 저장된 답변 메시지와 생성 중인 답변을 같은 모양으로 담아요 */
export interface SearchTurnAnswer {
  /** 답변 메시지 ID. 찾은 기록을 펼칠 때 주소에 실어요 */
  id: number;

  content: string;

  /** 생성 중인 답변은 `STREAMING`이에요 */
  status: SearchMessage["status"];

  /** 답변에 저장된 근거 문서. rank 오름차순이에요 */
  evidences: SearchMessage["evidences"];
}

/** 화면에 그려지는 대화 단위. 질문 하나와 그에 대한 답변 하나를 묶어요 */
export interface SearchTurn {
  /** 목록 key. 생성 중인 턴은 접수 전후로 같은 값이라 접수돼도 다시 그려지지 않아요 */
  key: string;

  /** 질문 메시지 ID. 실패한 답변을 다시 요청할 때 써요. 접수 전에 보낸 질문은 아직 없어 `null`이에요 */
  questionId: number | null;

  /** 질문 내용 */
  question: string;

  /** 아직 답변이 없으면(접수 대기 중이거나 답변 메시지가 없으면) `null`이에요 */
  answer: SearchTurnAnswer | null;
}
