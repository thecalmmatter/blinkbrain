# **Blink Install Overview**

Few things we need to do before we run the apprun.sh

1. check if ollama is locally installed
2. ollama model llama3.1 is installed
3. ollama should be running and available
4. Its a local app as of now which uses a bit of gpu and its collab version is in progress (due to gpu limits, local install is much smmoother)
5. OpenAI or LLama apis can be added to the code for easy consumption
6. More translation models can be added for more experimentation
7. Streamlit does have session issues so this app being a earlier version runs the training multiple times and it slows down the query, its a TODO list to perform tune this app better