package com.knot.backend.search.infrastructure.persistence;

import com.knot.backend.search.domain.model.SearchAnswerExecution;
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
            select e.answerId from SearchAnswerExecution e
            where e.state = 'RUNNING' and e.deadlineAt <= :now order by e.deadlineAt, e.answerId
            """)
    List<Long> findExpired(
            Instant now,
            Pageable pageable
    );
}
