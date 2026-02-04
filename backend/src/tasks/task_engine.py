"""
Task Engine for FlowLoop - Generates cognitive training tasks.

Supports:
- Arithmetic (addition, subtraction, multiplication, division)
- Algebra (solve for x)
- Memory (sequence recall - future)
- Pattern recognition (future)

Difficulty scales from 1-10 based on:
- Number range
- Number of operations
- Time pressure
"""

import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of cognitive tasks."""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    MEMORY = "memory"
    PATTERN = "pattern"


class Operation(str, Enum):
    """Math operations."""
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "×"
    DIVIDE = "÷"


@dataclass
class Task:
    """A cognitive training task."""
    id: str
    task_type: TaskType
    question: str
    answer: str | float
    difficulty: int  # 1-10
    time_limit_seconds: float
    hints: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class TaskGenerator:
    """Generates cognitive training tasks with adaptive difficulty."""
    
    def __init__(self):
        self._task_counter = 0
        self._generators: dict[TaskType, Callable[[int], Task]] = {
            TaskType.ARITHMETIC: self._generate_arithmetic,
            TaskType.ALGEBRA: self._generate_algebra,
        }
    
    def generate(self, task_type: TaskType, difficulty: int) -> Task:
        """
        Generate a task of the specified type and difficulty.
        
        Args:
            task_type: Type of task to generate
            difficulty: Difficulty level 1-10
        
        Returns:
            Generated Task
        """
        difficulty = max(1, min(10, difficulty))  # Clamp to 1-10
        
        generator = self._generators.get(task_type)
        if not generator:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return generator(difficulty)
    
    def _next_id(self) -> str:
        """Generate unique task ID."""
        self._task_counter += 1
        return f"task_{self._task_counter}"
    
    def _generate_arithmetic(self, difficulty: int) -> Task:
        """Generate arithmetic problem based on difficulty."""
        
        # Difficulty determines:
        # 1-2: Single digit add/subtract
        # 3-4: Double digit add/subtract
        # 5-6: Single digit multiply/divide
        # 7-8: Double digit multiply, triple digit add/subtract
        # 9-10: Mixed operations, larger numbers
        
        if difficulty <= 2:
            return self._arithmetic_simple(difficulty)
        elif difficulty <= 4:
            return self._arithmetic_double_digit(difficulty)
        elif difficulty <= 6:
            return self._arithmetic_multiply(difficulty)
        elif difficulty <= 8:
            return self._arithmetic_advanced(difficulty)
        else:
            return self._arithmetic_complex(difficulty)
    
    def _arithmetic_simple(self, difficulty: int) -> Task:
        """Single digit addition/subtraction."""
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        op = random.choice([Operation.ADD, Operation.SUBTRACT])
        
        if op == Operation.ADD:
            answer = a + b
            question = f"{a} + {b} = ?"
        else:
            # Ensure no negative results
            if a < b:
                a, b = b, a
            answer = a - b
            question = f"{a} - {b} = ?"
        
        return Task(
            id=self._next_id(),
            task_type=TaskType.ARITHMETIC,
            question=question,
            answer=answer,
            difficulty=difficulty,
            time_limit_seconds=15.0,
            hints=["Count on your fingers if needed"],
            metadata={"operation": op.value, "operands": [a, b]}
        )
    
    def _arithmetic_double_digit(self, difficulty: int) -> Task:
        """Double digit addition/subtraction."""
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        op = random.choice([Operation.ADD, Operation.SUBTRACT])
        
        if op == Operation.ADD:
            answer = a + b
            question = f"{a} + {b} = ?"
        else:
            if a < b:
                a, b = b, a
            answer = a - b
            question = f"{a} - {b} = ?"
        
        return Task(
            id=self._next_id(),
            task_type=TaskType.ARITHMETIC,
            question=question,
            answer=answer,
            difficulty=difficulty,
            time_limit_seconds=20.0,
            hints=["Break it into tens and ones"],
            metadata={"operation": op.value, "operands": [a, b]}
        )
    
    def _arithmetic_multiply(self, difficulty: int) -> Task:
        """Single digit multiplication/division."""
        a = random.randint(2, 12)
        b = random.randint(2, 12)
        op = random.choice([Operation.MULTIPLY, Operation.DIVIDE])
        
        if op == Operation.MULTIPLY:
            answer = a * b
            question = f"{a} × {b} = ?"
        else:
            # Ensure clean division
            product = a * b
            answer = a
            question = f"{product} ÷ {b} = ?"
        
        return Task(
            id=self._next_id(),
            task_type=TaskType.ARITHMETIC,
            question=question,
            answer=answer,
            difficulty=difficulty,
            time_limit_seconds=20.0,
            hints=["Think of multiplication tables"],
            metadata={"operation": op.value}
        )
    
    def _arithmetic_advanced(self, difficulty: int) -> Task:
        """Double digit multiplication or triple digit add/subtract."""
        if random.random() > 0.5:
            # Double digit times single digit
            a = random.randint(10, 99)
            b = random.randint(2, 9)
            answer = a * b
            question = f"{a} × {b} = ?"
            hint = "Break the larger number into parts"
        else:
            # Triple digit add/subtract
            a = random.randint(100, 999)
            b = random.randint(100, 999)
            if random.random() > 0.5:
                answer = a + b
                question = f"{a} + {b} = ?"
            else:
                if a < b:
                    a, b = b, a
                answer = a - b
                question = f"{a} - {b} = ?"
            hint = "Work digit by digit from right to left"
        
        return Task(
            id=self._next_id(),
            task_type=TaskType.ARITHMETIC,
            question=question,
            answer=answer,
            difficulty=difficulty,
            time_limit_seconds=30.0,
            hints=[hint],
            metadata={"level": "advanced"}
        )
    
    def _arithmetic_complex(self, difficulty: int) -> Task:
        """Mixed operations, order of operations."""
        a = random.randint(2, 20)
        b = random.randint(2, 10)
        c = random.randint(2, 10)
        
        patterns = [
            (f"{a} + {b} × {c}", a + b * c),
            (f"{a} × {b} + {c}", a * b + c),
            (f"{a * b} ÷ {b} + {c}", a + c),
            (f"({a} + {b}) × {c}", (a + b) * c),
        ]
        
        question, answer = random.choice(patterns)
        question += " = ?"
        
        return Task(
            id=self._next_id(),
            task_type=TaskType.ARITHMETIC,
            question=question,
            answer=answer,
            difficulty=difficulty,
            time_limit_seconds=45.0,
            hints=["Remember: multiplication before addition (PEMDAS)"],
            metadata={"level": "complex", "uses_order_of_operations": True}
        )
    
    def _generate_algebra(self, difficulty: int) -> Task:
        """Generate algebra problem (solve for x)."""
        
        if difficulty <= 3:
            # Simple: x + a = b
            a = random.randint(1, 10)
            x = random.randint(1, 10)
            b = x + a
            question = f"x + {a} = {b}"
            answer = x
            hint = f"Subtract {a} from both sides"
        elif difficulty <= 6:
            # Medium: ax = b or x/a = b
            a = random.randint(2, 10)
            x = random.randint(1, 10)
            if random.random() > 0.5:
                b = a * x
                question = f"{a}x = {b}"
                hint = f"Divide both sides by {a}"
            else:
                b = x  # x/a = b means x = a*b, we'll use simpler
                question = f"x ÷ {a} = {b}"
                answer = a * b
                hint = f"Multiply both sides by {a}"
            answer = x
        else:
            # Hard: ax + b = c
            a = random.randint(2, 5)
            b = random.randint(1, 10)
            x = random.randint(1, 10)
            c = a * x + b
            question = f"{a}x + {b} = {c}"
            answer = x
            hint = f"First subtract {b}, then divide by {a}"
        
        return Task(
            id=self._next_id(),
            task_type=TaskType.ALGEBRA,
            question=f"Solve for x: {question}",
            answer=answer,
            difficulty=difficulty,
            time_limit_seconds=30.0 + difficulty * 5,
            hints=[hint],
            metadata={"variable": "x"}
        )


class AdaptiveDifficultyController:
    """Controls difficulty based on user performance and cognitive state."""
    
    def __init__(
        self,
        initial_difficulty: int = 5,
        min_difficulty: int = 1,
        max_difficulty: int = 10,
    ):
        self.current_difficulty = initial_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        
        # Performance tracking
        self._recent_results: list[bool] = []  # True = correct
        self._recent_times: list[float] = []  # Response times
        self._window_size = 5
        
        # Thresholds
        self.increase_threshold = 0.8  # 80% correct = increase difficulty
        self.decrease_threshold = 0.4  # 40% correct = decrease difficulty
        
    def record_result(
        self,
        correct: bool,
        response_time: float,
        attention_score: float,
        frustration_score: float,
    ) -> int:
        """
        Record task result and update difficulty.
        
        Args:
            correct: Whether the answer was correct
            response_time: How long the user took
            attention_score: Current attention level (0-1)
            frustration_score: Current frustration level (0-1)
        
        Returns:
            New difficulty level
        """
        self._recent_results.append(correct)
        self._recent_times.append(response_time)
        
        # Keep window size
        if len(self._recent_results) > self._window_size:
            self._recent_results.pop(0)
            self._recent_times.pop(0)
        
        # Calculate performance
        if len(self._recent_results) >= 3:
            accuracy = sum(self._recent_results) / len(self._recent_results)
            
            # Adjust based on accuracy
            if accuracy >= self.increase_threshold:
                self._adjust(1, "High accuracy")
            elif accuracy <= self.decrease_threshold:
                self._adjust(-1, "Low accuracy")
        
        # Adjust based on cognitive state
        if frustration_score > 0.7:
            self._adjust(-1, "High frustration")
        elif attention_score < 0.3:
            self._adjust(-1, "Low attention")
        elif attention_score > 0.8 and frustration_score < 0.3:
            # Engaged and not frustrated - can push harder
            pass  # Already handled by accuracy
        
        return self.current_difficulty
    
    def _adjust(self, delta: int, reason: str) -> None:
        """Adjust difficulty with bounds checking."""
        new_difficulty = self.current_difficulty + delta
        new_difficulty = max(self.min_difficulty, min(self.max_difficulty, new_difficulty))
        
        if new_difficulty != self.current_difficulty:
            logger.info(f"Difficulty {self.current_difficulty} -> {new_difficulty}: {reason}")
            self.current_difficulty = new_difficulty
    
    def get_difficulty(self) -> int:
        """Get current difficulty level."""
        return self.current_difficulty
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.current_difficulty = 5
        self._recent_results.clear()
        self._recent_times.clear()


# Convenience functions
def create_task_generator() -> TaskGenerator:
    """Create a new task generator."""
    return TaskGenerator()


def create_difficulty_controller(initial: int = 5) -> AdaptiveDifficultyController:
    """Create a new adaptive difficulty controller."""
    return AdaptiveDifficultyController(initial_difficulty=initial)


if __name__ == "__main__":
    # Test
    generator = TaskGenerator()
    controller = AdaptiveDifficultyController()
    
    print("=== Task Engine Test ===\n")
    
    for i in range(10):
        difficulty = controller.get_difficulty()
        task = generator.generate(TaskType.ARITHMETIC, difficulty)
        
        print(f"Task {i+1} (Difficulty {difficulty})")
        print(f"  Question: {task.question}")
        print(f"  Answer: {task.answer}")
        print(f"  Time Limit: {task.time_limit_seconds}s")
        print()
        
        # Simulate response
        correct = random.random() > 0.3  # 70% success rate
        controller.record_result(
            correct=correct,
            response_time=10.0,
            attention_score=0.7,
            frustration_score=0.2,
        )
