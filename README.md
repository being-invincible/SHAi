# SHAi

![Empowering Agriculture Through Knowledge](https://github.com/being-invincible/SHAi/assets/86947956/4c61c548-b9d7-49de-af30-445ed9395695)


## Introduction:
SHAi, which stands for Sustainable Hydroponic AI, is a groundbreaking project aimed at revolutionizing hydroponic farming practices. Focused on sustainability, this domain-specific Large Language Model (LLM) is designed to cater to the diverse needs of hydroponic enthusiasts while aligning with eco-friendly agricultural methods.

## Key Features:

1. Comprehensive Guidance: SHAi provides comprehensive guidance for hydroponic farming, covering everything from basic principles to advanced industrial applications.
2. Sustainability Integration: Aligned with sustainable agricultural methods, SHAi promotes eco-conscious practices, contributing to a more environmentally friendly approach in hydroponics.
3. Domain-Specific Expertise: Tailored specifically for hydroponics, SHAi ensures that users receive specialized and relevant information, fostering success in their farming endeavors.
4. User-Friendly Interface: The SHAi interface is designed for accessibility, offering an intuitive platform for users at all skill levels, from beginners to experts.

## Hydroponics - A modern agricultural method
Hydroponics is a method of growing plants in water instead of soil. This can be done indoors or outdoors, and it can be used to grow a variety of different plants, including fruits, vegetables, and herbs.

Hydroponics is a sustainable method of growing plants that can help to improve food security and reduce the environmental impact of agriculture.

One of the main benefits of hydroponics is that it can be more efficient than traditional soil-based farming. This is because hydroponic systems can be more precisely controlled, which allows for better use of water and nutrients. In addition, hydroponic systems can be used to grow plants in areas where soil is not suitable for farming, such as in urban areas or in arid climates.

Hydroponics can also be used to grow plants year-round, which is not possible with traditional soil-based farming. This is because hydroponic systems can be controlled to provide the ideal growing conditions for plants, regardless of the time of year.

Another benefit of hydroponics is that it can reduce the amount of water and nutrients used to grow plants. This is because hydroponic systems recycle water and nutrients, which reduces the amount of waste produced. In addition, hydroponic systems can be used to grow plants with less water and nutrients than traditional soil-based farming methods.

Finally, hydroponics can help to reduce the amount of pollution produced by traditional farming methods. This is because hydroponic systems do not require the use of pesticides or herbicides, which can pollute the environment. In addition, hydroponic systems can be located close to urban areas, which reduces the need to transport food long distances.

Overall, hydroponics is a sustainable method of growing plants that can help to improve food security and reduce the environmental impact of agriculture.

## Problem Statement:
The absence of a comprehensive and accessible guide for hydroponic farming, spanning from home-based enthusiasts to industrial-scale practitioners, poses a significant barrier to widespread adoption. The risk of biased information and unsustainable agricultural practices obtained through internet searches, along with potential errors and human-induced hallucinations leading to untested theories, further complicates the reliable dissemination of knowledge in hydroponic farming. Bridging this gap is essential to encourage more people to embrace this innovative approach to agriculture.

## Proposed Solution:
In the dynamic landscape of hydroponic farming, accessibility to reliable information is crucial, spanning from home-based enthusiasts to industrial-grade practitioners. To address this, our project utilizes cutting-edge technologies, integrating Zilliz for storing vector embeddings derived from Google PALM Embeddings. This innovative approach harnesses web-scraped data stored in Github Markdown files to create a robust vector knowledge base.

## Methodology:
The SHAi project leverages the LlamaIndex framework to streamline the Retrieval Augmented Generation (RAG) process, enhancing the querying of the Zilliz vector database. With the assistance of LlamaIndex, context and prompts are efficiently extracted, enriching the information passed to Vertex AI's PaLM (Parameterized Language Model) for generating more prominent and relevant responses. This synergy between LlamaIndex and RAG ensures that the Language Model eliminates LLM hallucinations and accommodates a diverse audience, offering tailored guidance across proficiency levels in both home-based and industrial-grade hydroponic farming. The entire system is seamlessly integrated into a user-friendly Streamlit app, further enhancing accessibility and user interaction.

## Evaluation:
To gauge the effectiveness and relevance of our system, we employ Truera, an AI Observability platform. This platform serves as a comprehensive tool for LLM evaluation, ensuring that the responses generated by PALM align with the user's queries and contribute to the overall knowledge enhancement in hydroponic farming.

## Conclusion:
SHAi emerges as a powerful ally for those venturing into hydroponic farming, promoting sustainability and providing a wealth of knowledge in a user-friendly manner. As we advance towards a more sustainable future, SHAi stands at the forefront, guiding individuals on their hydroponic journey with expertise and eco-consciousness.
