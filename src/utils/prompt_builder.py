class Prompter:
    def __init__(
        self,
        prompt_template: dict[str, any],
    ):
        self.prompt_template = prompt_template

    def build_chat_prompt(
        self,
        query_text: str,
    ) -> list[dict[str, any]]:
        final_prompt = []
        if "system_prompt" in self.prompt_template.keys():
            content = self.prompt_template.get("system_prompt", "")
            final_prompt.append(
                {
                    "role": "system",
                    "content": content,
                }
            )
        if query_text is not None:
            final_prompt.append(
                {
                    "role": "user",
                    "content": query_text,
                }
            )
        else:
            raise ValueError("No query provided. Contact admin.")

        return final_prompt

    def reprompter(
        self,
        current_prompt: list,
        error_log: list,
    ):
        reprompt_content = ""
        error_count = 1
        if "reprompter" in self.prompt_template.keys():
            reprompter = self.prompt_template.get("reprompter", "")
            initial_message = reprompter.get("initial_message", "")
            reprompt_content = f"{reprompt_content}{initial_message}"
            for error in error_log:
                error_type = error.get("error_type", "")
                error_text = reprompter.get(error_type, "")
                error_text = error_text.format(
                    message=error.get("message", ""),
                    detail=error.get("detail", ""),
                )
                reprompt_content = f"{reprompt_content}\n{error_count}. {error_text}"
                error_count += 1
            current_prompt.append(
                {
                    "role": "user",
                    "content": reprompt_content,
                }
            )
        return current_prompt

        pass
