

convos_indexes = []
with open("movie_conversations.txt") as convos_list:
    for line in convos_list:
        conv = line[line.index("["):]
        conv = conv.replace("\'", "")
        conv = conv.replace("[", "")
        conv = conv.replace("]", "")
        conv = conv.replace("\n", "")
        conv = conv.replace(" ", "")
        conv = conv.split(",")
        convos_indexes.append(conv)

texts = {}
with open("movie_lines.txt", encoding = 'utf-8', errors = 'replace') as movie_lines:
    for line in movie_lines:
        conv_data = line.split("+++$+++")
        text = conv_data[-1].strip()
        text = text.replace("\n", "")
        conv_dict = {"text": text, "speaker": conv_data[-2]}
        l_index = conv_data[0].replace(" ", "")
        texts[l_index] = conv_dict

convs_texts = []
for conv in convos_indexes:
    last_speaker = ""
    this_conv = []
    for l_index in conv:
        speaker = texts[l_index]["speaker"]

        # check to see if the last speaker is the same as this speaker; just add the texts if this is the case
        # otherwise append to list of texts
        if speaker != last_speaker:
            this_conv.append(texts[l_index]["text"])
        else:
            this_conv[-1] = this_conv[-1] + " " + texts[l_index]["text"]


    convs_texts.append(this_conv)

# iterate through all of the conversation pairs, take first entry and it's next as the response,
# as long as there is a next entry
pairs = []      
for conv in convs_texts:
    for index in range(len(conv)-1):
        pairs.append([conv[index], conv[index + 1]])

# finally, write to a file
with open("train.txt", mode = "w", encoding = "utf-8") as outfile:
    for p in pairs:
        outfile.write(p[0] + "----$----")
        outfile.write(p[1] + "\n")
        outfile.write("----\n")

        
        
