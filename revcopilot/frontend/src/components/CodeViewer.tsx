import React from 'react';

interface CodeViewerProps {
    code: string;
    language: string;
}

const CodeViewer: React.FC<CodeViewerProps> = ({ code, language }) => {
    return (
        <div className="code-viewer">
            <pre>
                <code className={language}>
                    {code}
                </code>
            </pre>
        </div>
    );
};

export default CodeViewer;