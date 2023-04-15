with open("sentences_vejica.txt", "r") as input_file:
    with open("sentences.txt", "w") as output_file:
        for line in input_file:
            line = line.strip()
            # Skip sentences that we don't like (tweets and wrong comma positioning)
            if "¤" in line or "÷" in line or 'Janes' in line:
                continue
            parts = line.split("\t")
            if len(parts) > 1:

                # Remove sentences shorter than 5 words (will be hard to paraphrase)
                if len(parts[1].split()) < 5:
                    continue

                # Skip those that start with a lowercase letter
                if parts[1][0].isalpha() and not parts[1][0].isupper():
                    continue

                output_file.write(parts[1] + "\n")
