package com.knot.backend.search.infrastructure.persistence;

import com.knot.backend.search.domain.model.SearchAnswerExecution;
import com.knot.backend.search.domain.repository.SearchGenerationCandidate;
import jakarta.persistence.LockModeType;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;

interface SearchAnswerExecutionJpaRepository extends JpaRepository<SearchAnswerExecution, Long> {

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select e from SearchAnswerExecution e where e.answerId = :answerId")
    Optional<SearchAnswerExecution> findForUpdate(long answerId);

    @Query("select e.answerId from SearchAnswerExecution e where e.state = 'QUEUED' order by e.createdAt, e.answerId")
    List<Long> findQueued(Pageable pageable);

    @Query("""
            select e.answerId from SearchAnswerExecution e
            where e.state = 'QUEUED' and e.lookupOnly = :lookupOnly order by e.createdAt, e.answerId
            """)
    List<Long> findQueuedByLane(boolean lookupOnly, Pageable pageable);

    @Query("""
            select e.answerId from SearchAnswerExecution e
            where e.state = 'QUEUED' and e.lookupOnly = false and e.questionOnly = :questionOnly
            order by e.createdAt, e.answerId
            """)
    List<Long> findQueuedGenerationByLane(boolean questionOnly, Pageable pageable);

    @Query("""
            select new com.knot.backend.search.domain.repository.SearchGenerationCandidate(e.answerId, e.questionOnly)
            from SearchAnswerExecution e where e.state = 'QUEUED' and e.lookupOnly = false
            order by e.questionOnly desc, e.createdAt, e.answerId
            """)
    List<SearchGenerationCandidate> findQueuedGenerationByPriority(Pageable pageable);

    @Query("""
            select new com.knot.backend.search.domain.repository.SearchGenerationCandidate(e.answerId, e.questionOnly,
                case when e.synthesisQueuedAt <= :cutoff then true else false end)
            from SearchAnswerExecution e where e.state = 'QUEUED' and e.lookupOnly = false
            order by case when e.synthesisQueuedAt <= :cutoff then 0 when e.questionOnly = true then 1 else 2 end,
                case when e.synthesisQueuedAt <= :cutoff then e.synthesisQueuedAt else e.createdAt end, e.answerId
            """)
    List<SearchGenerationCandidate> findQueuedGenerationWithOverdueSynthesis(Instant cutoff, Pageable pageable);

    @Query("""
            select e.answerId from SearchAnswerExecution e
            where e.state = 'RUNNING' and e.deadlineAt <= :now order by e.deadlineAt, e.answerId
            """)
    List<Long> findExpired(
            Instant now,
            Pageable pageable
    );
}
