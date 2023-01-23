from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
import sys

def fact_check(question):
    task, model_id = 'text2text-generation', 'google/flan-t5-xxl'
    #task, model_id = 'text2text-generation', 'google/flan-t5-xl'
    #task, model_id = 'text-generation', 'bigscience/bloomz-7b1'
    #task, model_id = 'text2text-generation', 'bigscience/T0pp'
    llm = HuggingFacePipeline.from_model_id(task=task, model_id=model_id, model_kwargs=dict(device_map='auto', max_length=2048))
    #llm = OpenAI(temperature=0.7)
    #template = """Restate and answer the following question:\n\nQuestion: {question}\n\nRestatement and Answer:"""
    #template = """{question}\n\n"""
    template = """{question}"""
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    #template = """Here is a statement:\n\n{statement}\n\nMake a bullet point list of the assumptions you made when producing the above statement.\n\nBullet Point List:\n"""
    #template = """Here is a statement:\n\n{statement}\n\nMake a bullet point list of the assumptions you made when producing the above statement.\n\n"""
    #template = """Here is a statement:\n\n{statement}\n\nMake a bullet point list of the assumptions you made when producing the above statement."""
    template = """Here is a question and answer:\n\nQuestion: {question}\n\nAnswer: {answer}\n\nMake a bullet point list of the assumptions you made when producing the above answer."""
    prompt_template = PromptTemplate(input_variables=["question", "answer"], template=template)
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template, output_key='assumptions', verbose=True)

    #template = """Here is a bullet point list of assertions:\n\n{assertions}\n\nFor each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
    template = """Here is a bullet point list of assertions:\n\n{assumptions}\n\nFor each assertion, determine whether it is true or false. If it is false, explain why."""
    prompt_template = PromptTemplate(input_variables=["assumptions"], template=template)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template, output_key='assertions', verbose=True)

    template = """In light of the above facts, how would you answer the question '{}'""".format(question)
    template = """{facts}\n""" + template
    prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    assertion_building_chain = SequentialChain(input_variables=assumptions_chain.input_keys, output_variables=fact_checker_chain.output_keys, chains=[assumptions_chain, fact_checker_chain], verbose=True)

    answers = []
    assertions = []
    next_answer = question_chain.run(question)
    while next_answer not in answers:
        assertions.append(assertion_building_chain(dict(question=question, answer=next_answer))['assertions'])
        answers.append(next_answer)
        next_answer = answer_chain.run('\n'.join(assertions))
    return next_answer

if __name__=="__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "What type of mammal lays the biggest eggs?"
    print(question)
    answer = fact_check(question)
    print(answer)
