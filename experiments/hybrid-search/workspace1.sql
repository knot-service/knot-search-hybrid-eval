begin isolation level repeatable read read only;
set local statement_timeout = '15s';
select json_build_object(
  'workspaceId', 1,
  'documents', coalesce(json_agg(json_build_object(
    'id', d.id, 'meetingId', r.id, 'recordedAt', r.started_at,
    'title', d.title, 'topic', coalesce(d.topic, ''),
    'summary', coalesce(d.summary, ''), 'body', coalesce(d.content, '')
  ) order by d.id), '[]'::json)
)
from documents d
join recording_sessions r on r.id = d.recording_session_id and r.workspace_id = d.workspace_id
join workspaces w on w.id = d.workspace_id and w.deleted_at is null
where d.workspace_id = 1 and d.status in ('DRAFT','ARCHIVED');
commit;
