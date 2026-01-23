import React from 'react';
import Head from 'next/head';
import Navbar from '../components/Navbar';
import FileUpload from '../components/FileUpload';
import ResultsPanel from '../components/ResultsPanel';

const Home = () => {
    return (
        <div>
            <Head>
                <title>RevCopilot</title>
                <meta name="description" content="AI-Powered Reverse Engineering Assistant" />
                <link rel="icon" href="/favicon.ico" />
            </Head>
            <Navbar />
            <main className="flex flex-col items-center justify-center min-h-screen">
                <h1 className="text-4xl font-bold mb-4">Welcome to RevCopilot</h1>
                <FileUpload />
                <ResultsPanel />
            </main>
        </div>
    );
};

export default Home;