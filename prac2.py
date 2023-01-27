import aiml


kernel = aiml.Kernel()
kernel.learn("std-startup2.xml")
kernel.respond("load aiml b")


while True:
    input_text = input(">Human: ")
    response = kernel.respond(input_text)
    print(">Bot: "+response)


