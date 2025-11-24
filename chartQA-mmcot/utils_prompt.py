'''
Adapted from https://github.com/lupantech/ScienceQA
'''

from dataclasses import dataclass
from typing import List, Optional

def get_question_text(problem):
    # ScienceQA uses "question", ChartQA uses "query"
    if 'question' in problem:
        return problem['question']
    elif 'query' in problem:
        return problem['query']
    elif 'question_string' in problem:
        return problem['question_string']
    else:
        # Fallback for unknown structure
        return str(problem)

def get_context_text(problem, use_caption):
    # ScienceQA has "hint", ChartQA may have "table" or "caption"
    if 'hint' in problem:
        return problem['hint']
    elif 'context' in problem:
        return problem['context']
    elif 'table' in problem:
        return str(problem['table'])
    elif 'caption' in problem:
        return problem['caption']
    else:
        return ""




def get_choice_text(problem, options):
    """
    Builds a string like '(A) 2018 (B) 2019 (C) 2020' for multiple-choice questions.
    Handles any number of options or even open-ended questions gracefully.
    """

    # Ensure 'choices' exists and is a proper list
    choices = problem.get("choices", [])
    if not isinstance(choices, list) or len(choices) == 0:
        return ""  # no choices → open-ended QA

    # Clip available option letters to match choice count
    usable_options = options[:len(choices)]
    choice_list = []

    # Iterate safely over both lists together
    for letter, text in zip(usable_options, choices):
        # guard against empty or NaN values
        if text is None or str(text).strip() == "" or str(text).lower() == "nan":
            continue
        choice_list.append(f"({letter}) {text}")

    return " ".join(choice_list)



def get_origin_answer(problem, options):
    return problem['choices'][problem['answer']]

def get_answer(problem, options):
    """
    Returns the correct answer label for ScienceQA (integer index)
    or the literal answer string for datasets like ChartQA.
    """
    ans = problem.get("answer", "")

    # ScienceQA style: answer is index (int)
    if isinstance(ans, int) and ans < len(options):
        return options[ans]

    # ChartQA style: answer is a string (like "2016")
    if isinstance(ans, str):
        return ans

    # fallback
    return str(ans)


def get_lecture_text(problem):
    """
    Returns the lecture/explanatory text for ScienceQA.
    For custom datasets like ChartQA (which lack 'lecture'), returns an empty string.
    """
    lecture = problem.get("lecture", "")
    if not isinstance(lecture, str):
        lecture = str(lecture)
    return lecture.replace("\n", "\\n")



def get_solution_text(problem):
    """
    Returns the solution text (step-by-step explanation).
    For ChartQA or other custom datasets, fallback to an empty string or answer text.
    """
    solution = problem.get("solution", "")
    if not isinstance(solution, str):
        solution = str(solution)

    # if there's no solution field, try using answer as a proxy rationale
    if solution.strip() == "":
        if "answer" in problem:
            solution = f"The correct answer is {problem['answer']}."
        else:
            solution = ""

    return solution.replace("\n", "\\n")



def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True, WithOutput = False, curr_le_data=None):
        # Handle simple non-MCQ formats (like QA)
    if "-" not in format:
        if format == "QA":
            # Simple Question–Answer format (open-ended CoT)
            input = f"Question: {question}\nLet's think step by step.\n"
            if test_example:
                output = "Answer:"
            else:
                output = f"Answer: {answer}"
            text = input + output
            return text.strip()

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    elif input_format == "QM":
        input = f"Question: {question}\nOptions: {choice}\n"
    elif input_format == "QC":
        input = f"Question: {question}\nContext: {context}\n"
    elif input_format == "QCMG":
        if curr_le_data is not None:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    elif input_format == "CQMG":
        if curr_le_data is not None:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"
    elif input_format == "QCMA":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nAnswer: The answer is {answer}.\n"
    elif input_format == "QCA":
        input = f"Question: {question}\nContext: {context}\nAnswer: The answer is {answer}. \nBECAUSE:"

    # Outputs
    if test_example:
        if output_format == 'A':
            output = "Answer:"
        elif output_format == 'E':
            output = "Solution:"
        else:
            output = "Solution:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    elif output_format == 'LE':
        output = f"Solution: {lecture} {solution}."

    elif output_format == 'E':
        output = f"Solution: {solution}"
        
    
    if WithOutput:
        if output.endswith("BECAUSE:"):
            output = output.replace("BECAUSE:", "").strip()
        if output_format == 'E':
            text = input + f'Solution:'
        elif output_format == 'A':
            text = input + f'Answer:'
        else: 
            text = input + f'Solution:'
        text = text.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        return text, output
        
        
    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text


def build_prompt(problems, shot_qids, test_qid, args):

    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])


                # 🧩 Handle ChartQA-style problems (no choices, no context)
        if "choices" not in problems[qid]:
            question = problems[qid].get("question", "")
            context = problems[qid].get("context", "")
            choice = ""
            lecture = ""
            solution = problems[qid].get("rationale", "")
            answer = problems[qid].get("answer", "")

        train_example = create_one_example(args.prompt_format,
                                           question,
                                           context,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=False)
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=True)
    examples.append(test_example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input

def build_train_pair(problems, test_qid, args, curr_le_data=None):

    examples = []

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])
    answer_option = get_answer(problems[test_qid], args.options)
    answer = "(" + answer_option + ")"
    
        # 🧩 Handle ChartQA-style data
    if "choices" not in problems[test_qid]:
        question = problems[test_qid].get("question", "")
        context = problems[test_qid].get("context", "")
        choice = ""
        lecture = ""
        solution = problems[test_qid].get("rationale", "")
        answer = problems[test_qid].get("answer", "")

    test_example, target = create_one_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=False,WithOutput = True, curr_le_data=curr_le_data)
    examples.append(test_example)
    
    target = target.replace("Answer:", "").strip()
    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input, target

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    le_input_ids: List[List[int]]
    le_attention_mask: Optional[List[List[int]]]
    le_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]