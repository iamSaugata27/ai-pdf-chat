import express, { NextFunction, Request, RequestHandler, Response } from 'express';
import multer, { diskStorage } from 'multer';
import { unlinkSync } from 'fs';
import { config } from 'dotenv';
import { OpenAIEmbeddings } from "@langchain/openai";
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { HuggingFaceInference } from '@langchain/community/llms/hf';
import { InferenceClient } from "@huggingface/inference";
import { PineconeStore } from '@langchain/pinecone';
import { Pinecone } from "@pinecone-database/pinecone";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from '@langchain/core/documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { getPdfHash } from './utils/pdfHash';
import { isNamespaceEmpty } from './utils/isNamespaceEmpty';

config();

const app = express();
const port = 8500;

app.use(express.json());

const storage = diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        // console.log(file)
        cb(null, `${Date.now()}-${file.originalname}`);
    }
});

const upload = multer({ storage: storage });

// const embeddings = new OpenAIEmbeddings({
//     apiKey: process.env.OPENAI_API_KEY
// });
const embeddings = new HuggingFaceInferenceEmbeddings({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    apiKey: process.env.HuggingFace_API_Key
});
const model = new HuggingFaceInference({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    apiKey: process.env.HuggingFace_API_Key, // must include Inference API scope

});
const client = new InferenceClient(process.env.HuggingFace_API_Key);

const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY!
});
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME!);

app.post('/api/upload', upload.single('file'), async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    console.log('hiii')
    // const hash = await model.invoke("What is the capital of France?");
    // console.log('response from huggingface...', hash);
    const filePath = req.file?.path;
    if (!filePath) {
        res.status(400).send("No file uploaded.");
        return;
    }
    console.log(filePath)
    const hash = getPdfHash(filePath);
    console.log(hash);
    const shouldUpload = await isNamespaceEmpty(pineconeIndex, hash, embeddings);
    if (!shouldUpload) {
        unlinkSync(filePath);
        res.status(200).json({ message: "PDF already processed", namespace: hash });
        return;
    }
    const loader = new PDFLoader(filePath);
    const rawDocs = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    });
    const docs = await splitter.splitDocuments(rawDocs);
    await PineconeStore.fromDocuments(docs, embeddings, {
        pineconeIndex,
        namespace: hash
    });
    unlinkSync(filePath);
    next();
    res.status(200).send({ message: "PDF processed and stored", namespace: hash });
});

app.post("/api/ask", async (req: Request, res: Response): Promise<void> => {
    const { namespace, question } = req.body;

    if (!namespace || !question) {
        res.status(400).json({ error: "namespace and question are required" });
        return;
    }

    try {
        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex,
            namespace,
        });

        const retriever = vectorStore.asRetriever();
        // const retriever = {
        //     getrelevantDocuments: async (query:string) => {
        //         const embeddedQuery = await embeddings.embedQuery(query);
        //         const results = await pineconeIndex.query({
        //             vector: embeddedQuery,
        //             topK: 4,
        //             includeMetadata: true
        //         });
        //         return results.matches.map(m => new Document({ pageContent: <string>m.metadata?.text }));
        //     }
        // }

        const llm = new HuggingFaceInference({
            apiKey: process.env.HuggingFace_API_Key,
            model: 'tiiuae/falcon-7b-instruct' //'mistralai/Mistral-7B-Instruct-v0.1'    //
        });

        const prompt = ChatPromptTemplate.fromTemplate(`Use the following context to answer the user's question.
    Context:
    {context}

    Question:
    {input}`);
        const combineDocsChain = await createStuffDocumentsChain({ llm, prompt });
        const chain = await createRetrievalChain({ retriever, combineDocsChain });
        const response = await chain.invoke({ input: question });
        // const model = new ChatOpenAI({ temperature: 0 });
        // const chain = RetrievalQAChain.fromLLM(model, retriever);

        // const result = await chain.call({ query: question });

        res.status(200).json({ answer: response.answer });
    } catch (err) {
        console.error("Error processing question:", err);
        res.status(500).json({ error: "Failed to retrieve answer" });
    }
});

app.listen(port, () => {
    console.log(`PDF-AI-Agent API running at http://localhost:${port}`);
});