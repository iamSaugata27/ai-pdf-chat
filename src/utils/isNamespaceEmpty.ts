import { PineconeStore } from "@langchain/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';


export async function isNamespaceEmpty(
    pineconeIndex: any,
    namespace: string,
    embeddings: HuggingFaceInferenceEmbeddings): Promise<boolean> {
    try {
        const vectorStore = await PineconeStore.fromExistingIndex(embeddings,
            { pineconeIndex, namespace }
        );
        const results = vectorStore.similaritySearch("test", 1);
        return (await results).length === 0;
    } catch {
        return true;
    }
}