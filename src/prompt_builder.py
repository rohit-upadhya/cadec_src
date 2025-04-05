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
