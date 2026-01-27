import dspy

class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""

    problem = dspy.InputField()
    answer = dspy.OutputField()


class GenerateResponseWithReasoning(dspy.Signature):
    # https://claude.ai/share/01be74fe-62dd-4e4b-b948-32afbd69c5cc
    """Solve the problem and provide the answer in the correct format."""

    problem = dspy.InputField()
    reasoning = dspy.OutputField(prefix="Reasoning: Let's think step by step in order to", desc="${reasoning}")
    answer = dspy.OutputField()